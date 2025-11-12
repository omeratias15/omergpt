"""
Key Manager for omerGPT
AES-256 encryption for sensitive configuration
"""
import os
import base64
import logging
from pathlib import Path
from typing import Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

class KeyManager:
    """Manage encrypted keys and secrets"""

    def __init__(self, key_file: str = "C:/LLM/omerGPT/.master.key"):
        self.key_file = Path(key_file)
        self.env_file = Path("C:/LLM/omerGPT/.env")
        self.env_enc_file = Path("C:/LLM/omerGPT/.env.enc")

        self._master_key = None
        self._fernet = None

    def generate_master_key(self, password: Optional[str] = None) -> bytes:
        """Generate or derive master encryption key"""
        if password:
            # Derive key from password
            salt = b'omergpt_salt_v3'  # In production, use random salt
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        else:
            # Generate random key
            key = Fernet.generate_key()

        return key

    def save_master_key(self, key: bytes):
        """Save master key to file"""
        try:
            with open(self.key_file, 'wb') as f:
                f.write(key)

            # Set file permissions (Windows)
            if os.name == 'nt':
                import stat
                os.chmod(self.key_file, stat.S_IREAD | stat.S_IWRITE)

            logger.info(f"Master key saved to {self.key_file}")
        except Exception as e:
            logger.error(f"Error saving master key: {e}")
            raise

    def load_master_key(self) -> bytes:
        """Load master key from file"""
        try:
            if not self.key_file.exists():
                raise FileNotFoundError(f"Master key not found: {self.key_file}")

            with open(self.key_file, 'rb') as f:
                key = f.read()

            return key
        except Exception as e:
            logger.error(f"Error loading master key: {e}")
            raise

    def get_fernet(self) -> Fernet:
        """Get Fernet cipher instance"""
        if self._fernet is None:
            if self._master_key is None:
                self._master_key = self.load_master_key()
            self._fernet = Fernet(self._master_key)
        return self._fernet

    def encrypt_file(self, input_file: Path, output_file: Path):
        """Encrypt a file"""
        try:
            fernet = self.get_fernet()

            # Read plaintext
            with open(input_file, 'rb') as f:
                plaintext = f.read()

            # Encrypt
            ciphertext = fernet.encrypt(plaintext)

            # Write encrypted
            with open(output_file, 'wb') as f:
                f.write(ciphertext)

            logger.info(f"Encrypted {input_file} -> {output_file}")
        except Exception as e:
            logger.error(f"Error encrypting file: {e}")
            raise

    def decrypt_file(self, input_file: Path, output_file: Path):
        """Decrypt a file"""
        try:
            fernet = self.get_fernet()

            # Read ciphertext
            with open(input_file, 'rb') as f:
                ciphertext = f.read()

            # Decrypt
            plaintext = fernet.decrypt(ciphertext)

            # Write plaintext
            with open(output_file, 'wb') as f:
                f.write(plaintext)

            logger.info(f"Decrypted {input_file} -> {output_file}")
        except Exception as e:
            logger.error(f"Error decrypting file: {e}")
            raise

    def encrypt_env(self):
        """Encrypt .env file"""
        if not self.env_file.exists():
            logger.warning(f".env file not found: {self.env_file}")
            return

        self.encrypt_file(self.env_file, self.env_enc_file)
        logger.info("Environment file encrypted")

    def decrypt_env(self):
        """Decrypt .env.enc file"""
        if not self.env_enc_file.exists():
            logger.warning(f".env.enc file not found: {self.env_enc_file}")
            return

        self.decrypt_file(self.env_enc_file, self.env_file)
        logger.info("Environment file decrypted")

    def load_encrypted_env(self) -> dict:
        """Load and decrypt environment variables"""
        try:
            if not self.env_enc_file.exists():
                logger.warning("No encrypted env file found")
                return {}

            fernet = self.get_fernet()

            # Read and decrypt
            with open(self.env_enc_file, 'rb') as f:
                ciphertext = f.read()

            plaintext = fernet.decrypt(ciphertext)

            # Parse env variables
            env_vars = {}
            for line in plaintext.decode().split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()

            logger.info(f"Loaded {len(env_vars)} encrypted environment variables")
            return env_vars

        except Exception as e:
            logger.error(f"Error loading encrypted env: {e}")
            return {}

    def initialize_encryption(self, password: Optional[str] = None):
        """Initialize encryption system"""
        logger.info("Initializing encryption system...")

        # Generate master key
        if not self.key_file.exists():
            key = self.generate_master_key(password)
            self.save_master_key(key)
            self._master_key = key
            logger.info("Master key generated")
        else:
            logger.info("Master key already exists")

        # Encrypt .env if exists
        if self.env_file.exists() and not self.env_enc_file.exists():
            self.encrypt_env()

def main():
    """Initialize or test key manager"""
    logging.basicConfig(level=logging.INFO)

    manager = KeyManager()

    # Initialize (creates master key if not exists)
    manager.initialize_encryption()

    print("\nKey Manager initialized successfully!")
    print(f"Master key: {manager.key_file}")
    print(f"Encrypted env: {manager.env_enc_file}")

if __name__ == "__main__":
    main()
