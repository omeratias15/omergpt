"""
src/security/encrypt_env.py
AES-256 encryption and decryption for .env files and sensitive configuration.
Uses cryptography.fernet for symmetric encryption with secure key management.
Provides CLI interface for encrypting/decrypting environment files and API keys.
"""
import argparse
import base64
import getpass
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    Fernet = None

logger = logging.getLogger("omerGPT.security.encrypt_env")


class KeyManager:
    """
    Secure key management for encryption/decryption operations.
    Handles key generation, storage, and derivation from passwords.
    """
    
    def __init__(self):
        """Initialize key manager."""
        logger.info("KeyManager initialized")
    
    @staticmethod
    def generate_key() -> bytes:
        """
        Generate a new Fernet encryption key.
        
        Returns:
            32-byte encryption key (base64 encoded)
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography library required")
        
        key = Fernet.generate_key()
        logger.info("Generated new encryption key")
        return key
    
    @staticmethod
    def derive_key_from_password(
        password: str,
        salt: Optional[bytes] = None
    ) -> Tuple[bytes, bytes]:
        """
        Derive encryption key from password using PBKDF2.
        
        Args:
            password: User password
            salt: Salt for key derivation (generates new if None)
        
        Returns:
            Tuple of (encryption_key, salt)
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography library required")
        
        if salt is None:
            salt = os.urandom(16)
        
        # Use PBKDF2 for key derivation
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        
        logger.info("Derived encryption key from password")
        
        return key, salt
    
    @staticmethod
    def save_key(key: bytes, filepath: str):
        """
        Save encryption key to file.
        
        Args:
            key: Encryption key
            filepath: Output file path
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        with open(filepath, 'wb') as f:
            f.write(key)
        
        # Set restrictive permissions (Unix)
        try:
            os.chmod(filepath, 0o600)
        except:
            pass
        
        logger.info(f"Encryption key saved: {filepath}")
    
    @staticmethod
    def load_key(filepath: str) -> bytes:
        """
        Load encryption key from file.
        
        Args:
            filepath: Key file path
        
        Returns:
            Encryption key
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Key file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            key = f.read()
        
        logger.info(f"Encryption key loaded: {filepath}")
        
        return key
    
    @staticmethod
    def hash_key(key: bytes) -> str:
        """
        Generate hash of key for verification.
        
        Args:
            key: Encryption key
        
        Returns:
            SHA256 hash (hex string)
        """
        return hashlib.sha256(key).hexdigest()


class EnvEncryptor:
    """
    Encrypt and decrypt .env files and configuration data.
    Preserves key-value structure while encrypting values.
    """
    
    def __init__(self, key: bytes):
        """
        Initialize encryptor with encryption key.
        
        Args:
            key: Fernet encryption key
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography library required")
        
        self.cipher = Fernet(key)
        self.key_hash = KeyManager.hash_key(key)
        
        logger.info(f"EnvEncryptor initialized (key hash: {self.key_hash[:16]}...)")
    
    def encrypt_text(self, plaintext: str) -> str:
        """
        Encrypt plaintext string.
        
        Args:
            plaintext: Text to encrypt
        
        Returns:
            Encrypted text (base64 string)
        """
        encrypted = self.cipher.encrypt(plaintext.encode())
        return encrypted.decode()
    
    def decrypt_text(self, ciphertext: str) -> str:
        """
        Decrypt ciphertext string.
        
        Args:
            ciphertext: Encrypted text
        
        Returns:
            Decrypted plaintext
        """
        try:
            decrypted = self.cipher.decrypt(ciphertext.encode())
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Decryption failed - invalid key or corrupted data")
    
    def parse_env_file(self, filepath: str) -> Dict[str, str]:
        """
        Parse .env file into key-value dictionary.
        
        Args:
            filepath: Path to .env file
        
        Returns:
            Dictionary of environment variables
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        env_vars = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    env_vars[key] = value
                else:
                    logger.warning(f"Skipping invalid line {line_num}: {line}")
        
        logger.info(f"Parsed {len(env_vars)} variables from {filepath}")
        
        return env_vars
    
    def write_env_file(self, env_vars: Dict[str, str], filepath: str):
        """
        Write environment variables to .env file.
        
        Args:
            env_vars: Dictionary of environment variables
            filepath: Output file path
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Encrypted environment configuration\n")
            f.write(f"# Key hash: {self.key_hash[:16]}...\n")
            f.write(f"# Generated: {os.path.basename(__file__)}\n\n")
            
            for key, value in sorted(env_vars.items()):
                f.write(f'{key}="{value}"\n')
        
        # Set restrictive permissions
        try:
            os.chmod(filepath, 0o600)
        except:
            pass
        
        logger.info(f"Written {len(env_vars)} variables to {filepath}")
    
    def encrypt_env_file(self, input_path: str, output_path: str) -> int:
        """
        Encrypt all values in .env file.
        
        Args:
            input_path: Input .env file
            output_path: Output encrypted .env file
        
        Returns:
            Number of encrypted variables
        """
        # Parse input
        env_vars = self.parse_env_file(input_path)
        
        # Encrypt values
        encrypted_vars = {}
        
        for key, value in env_vars.items():
            encrypted_value = self.encrypt_text(value)
            encrypted_vars[key] = encrypted_value
        
        # Write output
        self.write_env_file(encrypted_vars, output_path)
        
        logger.info(f"Encrypted {len(encrypted_vars)} variables: {input_path} -> {output_path}")
        
        return len(encrypted_vars)
    
    def decrypt_env_file(self, input_path: str, output_path: str) -> int:
        """
        Decrypt all values in encrypted .env file.
        
        Args:
            input_path: Input encrypted .env file
            output_path: Output decrypted .env file
        
        Returns:
            Number of decrypted variables
        """
        # Parse input
        encrypted_vars = self.parse_env_file(input_path)
        
        # Decrypt values
        decrypted_vars = {}
        
        for key, encrypted_value in encrypted_vars.items():
            try:
                decrypted_value = self.decrypt_text(encrypted_value)
                decrypted_vars[key] = decrypted_value
            except Exception as e:
                logger.error(f"Failed to decrypt {key}: {e}")
                raise
        
        # Write output
        self.write_env_file(decrypted_vars, output_path)
        
        logger.info(f"Decrypted {len(decrypted_vars)} variables: {input_path} -> {output_path}")
        
        return len(decrypted_vars)
    
    def encrypt_json_file(self, input_path: str, output_path: str):
        """
        Encrypt JSON configuration file.
        
        Args:
            input_path: Input JSON file
            output_path: Output encrypted JSON file
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Encrypt entire JSON as string
        json_str = json.dumps(data, indent=2)
        encrypted = self.encrypt_text(json_str)
        
        # Write encrypted data
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(encrypted)
        
        logger.info(f"Encrypted JSON: {input_path} -> {output_path}")
    
    def decrypt_json_file(self, input_path: str, output_path: str):
        """
        Decrypt encrypted JSON configuration file.
        
        Args:
            input_path: Input encrypted file
            output_path: Output decrypted JSON file
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            encrypted = f.read()
        
        # Decrypt
        json_str = self.decrypt_text(encrypted)
        data = json.loads(json_str)
        
        # Write decrypted data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Decrypted JSON: {input_path} -> {output_path}")
    
    def encrypt_value(self, key: str, value: str) -> str:
        """
        Encrypt a single key-value pair (for interactive use).
        
        Args:
            key: Variable name (for logging)
            value: Value to encrypt
        
        Returns:
            Encrypted value
        """
        encrypted = self.encrypt_text(value)
        logger.info(f"Encrypted value for key: {key}")
        return encrypted
    
    def decrypt_value(self, key: str, encrypted_value: str) -> str:
        """
        Decrypt a single value (for interactive use).
        
        Args:
            key: Variable name (for logging)
            encrypted_value: Encrypted value
        
        Returns:
            Decrypted value
        """
        decrypted = self.decrypt_text(encrypted_value)
        logger.info(f"Decrypted value for key: {key}")
        return decrypted


class SecureEnvLoader:
    """
    Load encrypted environment variables at runtime.
    Automatically detects encrypted vs plaintext files.
    """
    
    def __init__(self, key_path: Optional[str] = None):
        """
        Initialize secure environment loader.
        
        Args:
            key_path: Path to encryption key file (None for plaintext)
        """
        self.key_path = key_path
        self.encryptor = None
        
        if key_path and os.path.exists(key_path):
            key = KeyManager.load_key(key_path)
            self.encryptor = EnvEncryptor(key)
            logger.info("SecureEnvLoader initialized with encryption key")
        else:
            logger.info("SecureEnvLoader initialized (plaintext mode)")
    
    def load_env(self, filepath: str = ".env") -> Dict[str, str]:
        """
        Load environment variables from file.
        Automatically decrypts if encryptor is configured.
        
        Args:
            filepath: Path to .env file
        
        Returns:
            Dictionary of environment variables
        """
        if not os.path.exists(filepath):
            logger.warning(f"Environment file not found: {filepath}")
            return {}
        
        # Parse file
        env_vars = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    env_vars[key] = value
        
        # Decrypt if encryptor available
        if self.encryptor:
            decrypted_vars = {}
            
            for key, value in env_vars.items():
                try:
                    decrypted_value = self.encryptor.decrypt_text(value)
                    decrypted_vars[key] = decrypted_value
                except Exception:
                    # Not encrypted, use as-is
                    decrypted_vars[key] = value
            
            env_vars = decrypted_vars
            logger.info(f"Loaded and decrypted {len(env_vars)} variables from {filepath}")
        else:
            logger.info(f"Loaded {len(env_vars)} plaintext variables from {filepath}")
        
        return env_vars
    
    def set_env_vars(self, env_vars: Dict[str, str]):
        """
        Set environment variables in current process.
        
        Args:
            env_vars: Dictionary of environment variables
        """
        for key, value in env_vars.items():
            os.environ[key] = value
        
        logger.info(f"Set {len(env_vars)} environment variables")


def main():
    """CLI entrypoint for encryption operations."""
    parser = argparse.ArgumentParser(
        description="Encrypt/decrypt .env files and configuration"
    )
    
    # Operation mode
    parser.add_argument(
        "--encrypt",
        type=str,
        help="Encrypt file (provide input path)"
    )
    parser.add_argument(
        "--decrypt",
        type=str,
        help="Decrypt file (provide input path)"
    )
    parser.add_argument(
        "--generate-key",
        action="store_true",
        help="Generate new encryption key"
    )
    parser.add_argument(
        "--encrypt-value",
        action="store_true",
        help="Encrypt a single value interactively"
    )
    parser.add_argument(
        "--decrypt-value",
        action="store_true",
        help="Decrypt a single value interactively"
    )
    
    # Key management
    parser.add_argument(
        "--key",
        type=str,
        default="keyfile.key",
        help="Path to encryption key file"
    )
    parser.add_argument(
        "--use-password",
        action="store_true",
        help="Derive key from password instead of key file"
    )
    parser.add_argument(
        "--salt-file",
        type=str,
        default="salt.bin",
        help="Path to salt file (for password-based encryption)"
    )
    
    # I/O
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["env", "json"],
        default="env",
        help="File format"
    )
    
    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    if not CRYPTOGRAPHY_AVAILABLE:
        print("Error: cryptography library required")
        print("Install: pip install cryptography")
        sys.exit(1)
    
    try:
        # Generate key
        if args.generate_key:
            key = KeyManager.generate_key()
            KeyManager.save_key(key, args.key)
            print(f"Generated encryption key: {args.key}")
            print(f"Key hash: {KeyManager.hash_key(key)}")
            return
        
        # Get or derive key
        if args.use_password:
            password = getpass.getpass("Enter password: ")
            
            # Load or generate salt
            if os.path.exists(args.salt_file):
                with open(args.salt_file, 'rb') as f:
                    salt = f.read()
                print(f"Loaded salt from: {args.salt_file}")
            else:
                key, salt = KeyManager.derive_key_from_password(password)
                with open(args.salt_file, 'wb') as f:
                    f.write(salt)
                print(f"Generated new salt: {args.salt_file}")
            
            key, _ = KeyManager.derive_key_from_password(password, salt)
        else:
            if not os.path.exists(args.key):
                print(f"Error: Key file not found: {args.key}")
                print("Generate key with: --generate-key")
                sys.exit(1)
            
            key = KeyManager.load_key(args.key)
        
        # Initialize encryptor
        encryptor = EnvEncryptor(key)
        
        # Encrypt file
        if args.encrypt:
            input_path = args.encrypt
            output_path = args.output or f"{input_path}.encrypted"
            
            if args.format == "env":
                count = encryptor.encrypt_env_file(input_path, output_path)
                print(f"Encrypted {count} variables: {input_path} -> {output_path}")
            elif args.format == "json":
                encryptor.encrypt_json_file(input_path, output_path)
                print(f"Encrypted JSON: {input_path} -> {output_path}")
        
        # Decrypt file
        elif args.decrypt:
            input_path = args.decrypt
            output_path = args.output or input_path.replace(".encrypted", ".decrypted")
            
            if args.format == "env":
                count = encryptor.decrypt_env_file(input_path, output_path)
                print(f"Decrypted {count} variables: {input_path} -> {output_path}")
            elif args.format == "json":
                encryptor.decrypt_json_file(input_path, output_path)
                print(f"Decrypted JSON: {input_path} -> {output_path}")
        
        # Encrypt value
        elif args.encrypt_value:
            key_name = input("Enter key name: ")
            value = getpass.getpass("Enter value to encrypt: ")
            
            encrypted = encryptor.encrypt_value(key_name, value)
            print(f"\nEncrypted value for {key_name}:")
            print(encrypted)
        
        # Decrypt value
        elif args.decrypt_value:
            key_name = input("Enter key name: ")
            encrypted_value = input("Enter encrypted value: ")
            
            decrypted = encryptor.decrypt_value(key_name, encrypted_value)
            print(f"\nDecrypted value for {key_name}:")
            print(decrypted)
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)


# Test block
if __name__ == "__main__":
    if "--test" in sys.argv:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        
        print("Testing EnvEncryptor...")
        print(f"cryptography available: {CRYPTOGRAPHY_AVAILABLE}\n")
        
        if not CRYPTOGRAPHY_AVAILABLE:
            print("Error: cryptography library required")
            print("Install: pip install cryptography")
            sys.exit(1)
        
        # Test 1: Generate key
        print("1. Testing key generation...")
        key = KeyManager.generate_key()
        print(f"   Generated key (length: {len(key)} bytes)")
        print(f"   Key hash: {KeyManager.hash_key(key)[:32]}...")
        print()
        
        # Test 2: Save and load key
        print("2. Testing key save/load...")
        test_key_path = "test_keyfile.key"
        KeyManager.save_key(key, test_key_path)
        loaded_key = KeyManager.load_key(test_key_path)
        assert key == loaded_key, "Key mismatch!"
        print(f"   Key saved and loaded successfully")
        print()
        
        # Test 3: Password-based key derivation
        print("3. Testing password-based key derivation...")
        password = "test_password_123"
        derived_key, salt = KeyManager.derive_key_from_password(password)
        print(f"   Derived key from password")
        print(f"   Salt length: {len(salt)} bytes")
        
        # Verify deterministic derivation
        derived_key2, _ = KeyManager.derive_key_from_password(password, salt)
        assert derived_key == derived_key2, "Key derivation not deterministic!"
        print(f"   Key derivation is deterministic")
        print()
        
        # Test 4: Initialize encryptor
        print("4. Testing encryptor initialization...")
        encryptor = EnvEncryptor(key)
        print(f"   Encryptor initialized")
        print(f"   Key hash: {encryptor.key_hash[:32]}...")
        print()
        
        # Test 5: Encrypt and decrypt text
        print("5. Testing text encryption/decryption...")
        plaintext = "This is a secret message!"
        encrypted = encryptor.encrypt_text(plaintext)
        decrypted = encryptor.decrypt_text(encrypted)
        
        print(f"   Plaintext: {plaintext}")
        print(f"   Encrypted: {encrypted[:50]}...")
        print(f"   Decrypted: {decrypted}")
        assert plaintext == decrypted, "Text mismatch!"
        print()
        
        # Test 6: Create test .env file
        print("6. Testing .env file encryption...")
        test_env_path = "test.env"
        
        with open(test_env_path, 'w') as f:
            f.write("# Test environment file\n")
            f.write('API_KEY="sk-test-1234567890"\n')
            f.write('DATABASE_URL="postgresql://user:pass@localhost/db"\n')
            f.write('SECRET_TOKEN="abc123def456"\n')
            f.write('DEBUG="true"\n')
        
        print(f"   Created test .env file with 4 variables")
        print()
        
        # Test 7: Encrypt .env file
        print("7. Testing .env file encryption...")
        encrypted_env_path = "test.env.encrypted"
        count = encryptor.encrypt_env_file(test_env_path, encrypted_env_path)
        print(f"   Encrypted {count} variables")
        
        # Show encrypted content
        with open(encrypted_env_path, 'r') as f:
            lines = f.readlines()[:6]  # First 6 lines
            print("   Encrypted file preview:")
            for line in lines:
                print(f"     {line.rstrip()}")
        print()
        
        # Test 8: Decrypt .env file
        print("8. Testing .env file decryption...")
        decrypted_env_path = "test.env.decrypted"
        count = encryptor.decrypt_env_file(encrypted_env_path, decrypted_env_path)
        print(f"   Decrypted {count} variables")
        
        # Verify round-trip
        original = encryptor.parse_env_file(test_env_path)
        decrypted_vars = encryptor.parse_env_file(decrypted_env_path)
        
        assert original == decrypted_vars, "Round-trip failed!"
        print(f"   Round-trip verification successful")
        print()
        
        # Test 9: JSON encryption
        print("9. Testing JSON encryption...")
        test_json_path = "test_config.json"
        
        test_config = {
            "api_keys": {
                "openai": "sk-test-openai",
                "binance": "api-test-binance",
            },
            "database": {
                "host": "localhost",
                "password": "secret123",
            },
        }
        
        with open(test_json_path, 'w') as f:
            json.dump(test_config, f, indent=2)
        
        encrypted_json_path = "test_config.json.encrypted"
        encryptor.encrypt_json_file(test_json_path, encrypted_json_path)
        print(f"   Encrypted JSON file")
        
        decrypted_json_path = "test_config.json.decrypted"
        encryptor.decrypt_json_file(encrypted_json_path, decrypted_json_path)
        print(f"   Decrypted JSON file")
        
        with open(decrypted_json_path, 'r') as f:
            decrypted_config = json.load(f)
        
        assert test_config == decrypted_config, "JSON round-trip failed!"
        print(f"   JSON round-trip verification successful")
        print()
        
        # Test 10: SecureEnvLoader
        print("10. Testing SecureEnvLoader...")
        loader = SecureEnvLoader(key_path=test_key_path)
        env_vars = loader.load_env(encrypted_env_path)
        
        print(f"   Loaded {len(env_vars)} variables")
        print(f"   Variables: {list(env_vars.keys())}")
        
        assert env_vars["API_KEY"] == "sk-test-1234567890", "Value mismatch!"
        print(f"   Values decrypted correctly")
        print()
        
        # Cleanup
        print("11. Cleaning up test files...")
        for filepath in [
            test_key_path,
            test_env_path,
            encrypted_env_path,
            decrypted_env_path,
            test_json_path,
            encrypted_json_path,
            decrypted_json_path,
        ]:
            if os.path.exists(filepath):
                os.remove(filepath)
        print(f"   Removed test files")
        print()
        
        print("All tests passed successfully!")
        print("\nCLI Usage Examples:")
        print("  # Generate key")
        print("  python src/security/encrypt_env.py --generate-key --key mykey.key")
        print()
        print("  # Encrypt .env file")
        print("  python src/security/encrypt_env.py --encrypt .env --key mykey.key --output .env.encrypted")
        print()
        print("  # Decrypt .env file")
        print("  python src/security/encrypt_env.py --decrypt .env.encrypted --key mykey.key --output .env")
        print()
        print("  # Use password instead of key file")
        print("  python src/security/encrypt_env.py --encrypt .env --use-password --output .env.encrypted")
        print()
        print("  # Encrypt single value")
        print("  python src/security/encrypt_env.py --encrypt-value --key mykey.key")
    else:
        main()
