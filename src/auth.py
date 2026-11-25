import sqlite3
import os
import hashlib
import secrets
import hmac
from pathlib import Path
from datetime import datetime, timedelta


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = os.getenv("FINLYTIC_DATA_DIR") or str(PROJECT_ROOT / "data")
AUTH_DATABASE_PATH = os.path.join(DATA_DIR, "auth.db")

# PBKDF2-HMAC-SHA256 work factor (OWASP 2023 guidance: >= 600,000).
PBKDF2_ITERATIONS = 600_000
# Hashes created before versioning used "salt:hash" with this work factor.
LEGACY_PBKDF2_ITERATIONS = 100_000
HASH_SCHEME = "pbkdf2_sha256"

RESET_CODE_TTL_MINUTES = 10
MAX_RESET_ATTEMPTS = 5


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_auth_connection():
    os.makedirs(DATA_DIR, exist_ok=True)
    return sqlite3.connect(AUTH_DATABASE_PATH, timeout=30)


# ============================================================
# INITIALIZE AUTH DATABASE
# ============================================================

def init_auth_db():
    connection = get_auth_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            reset_code TEXT,
            reset_code_expires TEXT,
            reset_attempts INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # --------------------------------------------------------
    # Migration for an existing auth.db
    # --------------------------------------------------------

    try:
        cursor.execute(
            "ALTER TABLE users ADD COLUMN reset_code TEXT"
        )
    except sqlite3.OperationalError:
        # Column already exists
        pass

    try:
        cursor.execute(
            "ALTER TABLE users ADD COLUMN reset_code_expires TEXT"
        )
    except sqlite3.OperationalError:
        # Column already exists
        pass

    try:
        cursor.execute(
            "ALTER TABLE users ADD COLUMN reset_attempts INTEGER NOT NULL DEFAULT 0"
        )
    except sqlite3.OperationalError:
        pass

    connection.commit()
    connection.close()


# ============================================================
# PASSWORD HASHING
# ============================================================

def hash_password(password, iterations=PBKDF2_ITERATIONS):
    """Hash a password with PBKDF2-HMAC-SHA256 and a random 16-byte salt.

    Format: ``pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>``. Storing the
    work factor with the hash lets it be raised later without breaking
    existing accounts.
    """
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return f"{HASH_SCHEME}${iterations}${salt.hex()}${digest.hex()}"


def _parse_hash(stored_password):
    """Return (iterations, salt, digest) for current and legacy formats."""
    if stored_password.startswith(HASH_SCHEME + "$"):
        _, iterations, salt_hex, hash_hex = stored_password.split("$")
        return int(iterations), bytes.fromhex(salt_hex), bytes.fromhex(hash_hex)
    salt_hex, hash_hex = stored_password.split(":")
    return (
        LEGACY_PBKDF2_ITERATIONS,
        bytes.fromhex(salt_hex),
        bytes.fromhex(hash_hex),
    )


def verify_password(password, stored_password):
    """Constant-time verification of current or legacy hash formats."""
    try:
        iterations, salt, expected = _parse_hash(stored_password)
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(candidate, expected)
    except (ValueError, TypeError, AttributeError):
        return False


def needs_rehash(stored_password):
    """True when a stored hash uses an older scheme or a lower work factor."""
    try:
        if not stored_password.startswith(HASH_SCHEME + "$"):
            return True
        return int(stored_password.split("$")[1]) < PBKDF2_ITERATIONS
    except (ValueError, IndexError, AttributeError):
        return True


# ============================================================
# EMAIL VALIDATION
# ============================================================

def validate_email(email):
    email = email.strip().lower()

    if not email:
        return False

    if email.count("@") != 1 or any(char.isspace() for char in email):
        return False

    local, domain = email.split("@", 1)
    if not local or "." not in domain:
        return False

    if domain.startswith(".") or domain.endswith(".") or ".." in domain:
        return False

    return True


# ============================================================
# REGISTER USER
# ============================================================

def register_user(name, email, password):

    init_auth_db()

    name = name.strip()
    email = email.strip().lower()

    # Validate name
    if not name:
        return (
            False,
            "Name is required.",
            None,
        )

    # Validate email
    if not validate_email(email):
        return (
            False,
            "Please enter a valid email address.",
            None,
        )

    # Validate password
    if len(password) < 8:
        return (
            False,
            "Password must contain at least 8 characters.",
            None,
        )

    password_hash = hash_password(password)

    connection = get_auth_connection()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO users
            (
                name,
                email,
                password_hash
            )
            VALUES (?, ?, ?)
            """,
            (
                name,
                email,
                password_hash,
            ),
        )

        connection.commit()

        user_id = cursor.lastrowid

        return (
            True,
            "Account created successfully.",
            user_id,
        )

    except sqlite3.IntegrityError:

        return (
            False,
            "An account with this email already exists.",
            None,
        )

    except Exception as exc:

        return (
            False,
            f"Registration failed: {exc}",
            None,
        )

    finally:
        connection.close()


# ============================================================
# AUTHENTICATE USER
# ============================================================

def authenticate_user(email, password):

    init_auth_db()

    email = email.strip().lower()

    connection = get_auth_connection()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            SELECT
                id,
                name,
                email,
                password_hash
            FROM users
            WHERE email = ?
            """,
            (email,),
        )

        user = cursor.fetchone()

        if user is None:

            return {
                "success": False,
                "message": "Invalid email or password.",
            }

        user_id = user[0]
        name = user[1]
        stored_email = user[2]
        stored_password = user[3]

        if not verify_password(
            password,
            stored_password,
        ):

            return {
                "success": False,
                "message": "Invalid email or password.",
            }

        if needs_rehash(stored_password):
            # Transparently upgrade legacy / weaker hashes after a good login.
            cursor.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (hash_password(password), user_id),
            )
            connection.commit()

        return {
            "success": True,
            "user_id": user_id,
            "name": name,
            "email": stored_email,
        }

    finally:
        connection.close()


# ============================================================
# GET USER BY ID
# ============================================================

def get_user_by_id(user_id):

    init_auth_db()

    connection = get_auth_connection()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            SELECT
                id,
                name,
                email,
                created_at
            FROM users
            WHERE id = ?
            """,
            (int(user_id),),
        )

        user = cursor.fetchone()

        if user is None:
            return None

        return {
            "user_id": user[0],
            "name": user[1],
            "email": user[2],
            "created_at": user[3],
        }

    finally:
        connection.close()


# ============================================================
# CHECK IF EMAIL EXISTS
# ============================================================

def email_exists(email):

    init_auth_db()

    email = email.strip().lower()

    connection = get_auth_connection()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            SELECT 1
            FROM users
            WHERE email = ?
            LIMIT 1
            """,
            (email,),
        )

        return cursor.fetchone() is not None

    finally:
        connection.close()


# ============================================================
# PASSWORD RESET CODES
# ============================================================

def _hash_reset_code(email, code):
    """Hash a reset code so a leaked auth.db does not expose live codes."""
    return hashlib.sha256(f"{email}:{code}".encode("utf-8")).hexdigest()


# ============================================================
# CREATE PASSWORD RESET CODE
# ============================================================

def create_password_reset_code(email):

    init_auth_db()

    email = email.strip().lower()

    # Validate email
    if not validate_email(email):

        return (
            False,
            "Please enter a valid email address.",
            None,
        )

    connection = get_auth_connection()
    cursor = connection.cursor()

    try:

        # Check whether account exists
        cursor.execute(
            """
            SELECT id
            FROM users
            WHERE email = ?
            """,
            (email,),
        )

        user = cursor.fetchone()

        if user is None:

            return (
                False,
                "No account found with this email address.",
                None,
            )

        # Generate a 6-digit reset code
        reset_code = str(
            secrets.randbelow(900000) + 100000
        )

        # Code valid for 10 minutes
        expires_at = (
            datetime.now()
            + timedelta(minutes=RESET_CODE_TTL_MINUTES)
        ).isoformat()

        cursor.execute(
            """
            UPDATE users
            SET
                reset_code = ?,
                reset_code_expires = ?,
                reset_attempts = 0
            WHERE email = ?
            """,
            (
                _hash_reset_code(email, reset_code),
                expires_at,
                email,
            ),
        )

        connection.commit()

        return (
            True,
            "Reset code generated successfully.",
            reset_code,
        )

    except Exception as exc:

        return (
            False,
            f"Unable to generate reset code: {exc}",
            None,
        )

    finally:
        connection.close()


# ============================================================
# RESET PASSWORD
# ============================================================

def reset_password(
    email,
    reset_code,
    new_password,
):

    init_auth_db()

    email = email.strip().lower()
    reset_code = reset_code.strip()

    # Validate new password
    if len(new_password) < 8:

        return (
            False,
            "Password must contain at least 8 characters.",
        )

    connection = get_auth_connection()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            SELECT
                reset_code,
                reset_code_expires,
                reset_attempts
            FROM users
            WHERE email = ?
            """,
            (email,),
        )

        user = cursor.fetchone()

        if user is None:

            return (
                False,
                "Account not found.",
            )

        stored_code = user[0]
        expires_at = user[1]

        # Check code exists
        if not stored_code:

            return (
                False,
                "No active reset code found.",
            )

        attempts = user[2] or 0
        if attempts >= MAX_RESET_ATTEMPTS:
            cursor.execute(
                "UPDATE users SET reset_code = NULL, reset_code_expires = NULL "
                "WHERE email = ?",
                (email,),
            )
            connection.commit()
            return (
                False,
                "Too many incorrect attempts. Request a new reset code.",
            )

        # Check code matches (constant-time comparison of hashes)
        if not hmac.compare_digest(
            stored_code, _hash_reset_code(email, reset_code)
        ):
            cursor.execute(
                "UPDATE users SET reset_attempts = reset_attempts + 1 "
                "WHERE email = ?",
                (email,),
            )
            connection.commit()
            return (
                False,
                "Invalid reset code.",
            )

        # Check expiration exists
        if not expires_at:

            return (
                False,
                "Reset code has expired.",
            )

        # Check expiration
        try:

            expiration_time = datetime.fromisoformat(
                expires_at
            )

        except ValueError:

            return (
                False,
                "Invalid reset code expiration.",
            )

        if datetime.now() > expiration_time:

            return (
                False,
                "Reset code has expired.",
            )

        # Create new password hash
        new_password_hash = hash_password(
            new_password
        )

        # Update password and remove reset code
        cursor.execute(
            """
            UPDATE users
            SET
                password_hash = ?,
                reset_code = NULL,
                reset_code_expires = NULL,
                reset_attempts = 0
            WHERE email = ?
            """,
            (
                new_password_hash,
                email,
            ),
        )

        connection.commit()

        return (
            True,
            "Password reset successfully.",
        )

    except Exception as exc:

        connection.rollback()

        return (
            False,
            f"Password reset failed: {exc}",
        )

    finally:
        connection.close()