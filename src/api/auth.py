"""
JWT Authentication for omerGPT API
"""
import os
import jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "omergpt-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt

def verify_token(token: str) -> dict:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """Extract and verify current user from token"""
    token = credentials.credentials
    payload = verify_token(token)
    return payload

def generate_api_token(username: str = "omergpt_user") -> str:
    """Generate a new API token"""
    token_data = {
        "sub": username,
        "iat": datetime.utcnow()
    }
    token = create_access_token(token_data)
    return token

# For testing/development
if __name__ == "__main__":
    # Generate a test token
    test_token = generate_api_token("test_user")
    print(f"Test Token: {test_token}")

    # Verify it
    try:
        payload = verify_token(test_token)
        print(f"Token Valid: {payload}")
    except Exception as e:
        print(f"Token Invalid: {e}")
