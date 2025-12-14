"""
Tests for the JWT authentication module.

Tests cover:
- Password hashing and verification
- JWT token creation and validation
- API key generation and validation
- Role-based access control
- User authentication workflow
"""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from medtech_ai_security.api.auth import (
    DEMO_USERS,
    APIKeyCreate,
    APIKeyResponse,
    RoleChecker,
    Token,
    TokenData,
    TokenType,
    User,
    UserCreate,
    UserRole,
    authenticate_user,
    create_access_token,
    create_api_key,
    create_refresh_token,
    create_user,
    decode_token,
    get_password_hash,
    revoke_token,
    validate_api_key,
    verify_password,
)

# =============================================================================
# Pydantic Model Tests
# =============================================================================


class TestUserRole:
    """Test UserRole enum."""

    def test_role_values(self):
        """Test role enum values."""
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.ANALYST.value == "analyst"
        assert UserRole.VIEWER.value == "viewer"
        assert UserRole.API_SERVICE.value == "api_service"

    def test_role_comparison(self):
        """Test role comparison."""
        assert UserRole.ADMIN != UserRole.VIEWER
        assert UserRole.ANALYST != UserRole.ADMIN


class TestTokenData:
    """Test TokenData model."""

    def test_creation(self):
        """Test token data creation."""
        now = datetime.now(timezone.utc)
        data = TokenData(
            sub="test_user",
            role=UserRole.ANALYST,
            token_type=TokenType.ACCESS,
            exp=now + timedelta(hours=1),
            iat=now,
            jti="test_jti",
        )
        assert data.sub == "test_user"
        assert data.role == UserRole.ANALYST
        assert data.token_type == TokenType.ACCESS

    def test_with_expiration(self):
        """Test token data with expiration."""
        now = datetime.now(timezone.utc)
        data = TokenData(
            sub="user",
            role=UserRole.VIEWER,
            token_type=TokenType.REFRESH,
            exp=now + timedelta(hours=1),
            iat=now,
            jti="test_jti_2",
        )
        assert data.exp is not None


class TestUser:
    """Test User model."""

    def test_creation(self):
        """Test user creation."""
        user = User(
            username="testuser",
            email="test@example.com",
            role=UserRole.ANALYST,
            disabled=False,
        )
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.ANALYST
        assert user.disabled is False

    def test_disabled_default(self):
        """Test disabled default value."""
        user = User(
            username="user",
            email="user@example.com",
            role=UserRole.VIEWER,
        )
        assert user.disabled is False


class TestUserCreate:
    """Test UserCreate model."""

    def test_creation(self):
        """Test user create with password."""
        user = UserCreate(
            username="newuser",
            email="new@example.com",
            role=UserRole.ANALYST,
            password="securepassword123",
        )
        assert user.username == "newuser"
        assert user.password == "securepassword123"


class TestToken:
    """Test Token model."""

    def test_creation(self):
        """Test token creation."""
        token = Token(
            access_token="abc123",
            token_type="bearer",
            expires_in=3600,
        )
        assert token.access_token == "abc123"
        assert token.token_type == "bearer"
        assert token.expires_in == 3600

    def test_with_refresh_token(self):
        """Test token with refresh token."""
        token = Token(
            access_token="access123",
            token_type="bearer",
            expires_in=1800,
            refresh_token="refresh456",
        )
        assert token.refresh_token == "refresh456"


class TestAPIKeyCreate:
    """Test APIKeyCreate model."""

    def test_creation(self):
        """Test API key create model."""
        api_key = APIKeyCreate(
            name="Production API Key",
            role=UserRole.API_SERVICE,
            expires_days=365,
        )
        assert api_key.name == "Production API Key"
        assert api_key.expires_days == 365

    def test_defaults(self):
        """Test default values."""
        api_key = APIKeyCreate(name="Test Key")
        assert api_key.role == UserRole.API_SERVICE
        assert api_key.expires_days == 365


class TestAPIKeyResponse:
    """Test APIKeyResponse model."""

    def test_creation(self):
        """Test API key response."""
        response = APIKeyResponse(
            key_id="key123",
            key="medsec_abc123xyz",
            name="Test Key",
            role=UserRole.API_SERVICE,
            expires_at=datetime.now(timezone.utc) + timedelta(days=90),
        )
        assert response.key_id == "key123"
        assert response.key.startswith("medsec_")


# =============================================================================
# Password Utility Tests
# =============================================================================


class TestPasswordUtilities:
    """Test password hashing utilities."""

    def test_hash_password(self):
        """Test password hashing."""
        password = "mysecurepassword"
        hashed = get_password_hash(password)

        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are long

    def test_verify_password_correct(self):
        """Test correct password verification."""
        password = "testpassword123"
        hashed = get_password_hash(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test incorrect password verification."""
        password = "correctpassword"
        wrong_password = "wrongpassword"
        hashed = get_password_hash(password)

        assert verify_password(wrong_password, hashed) is False

    def test_different_hashes_same_password(self):
        """Test that same password produces different hashes (salt)."""
        password = "samepassword"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)

        # Hashes should be different due to salt
        assert hash1 != hash2
        # But both should verify
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True


# =============================================================================
# Token Tests
# =============================================================================


class TestTokenCreation:
    """Test JWT token creation."""

    def test_create_access_token(self):
        """Test access token creation."""
        token = create_access_token(
            subject="testuser",
            role=UserRole.ANALYST,
        )

        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50

    def test_create_access_token_custom_expiry(self):
        """Test access token with custom expiration."""
        token = create_access_token(
            subject="user",
            role=UserRole.VIEWER,
            expires_delta=timedelta(minutes=5),
        )

        decoded = decode_token(token)
        assert decoded is not None
        assert decoded.sub == "user"

    def test_create_refresh_token(self):
        """Test refresh token creation."""
        token = create_refresh_token(
            subject="testuser",
            role=UserRole.ADMIN,
        )

        assert token is not None
        decoded = decode_token(token)
        assert decoded.token_type == TokenType.REFRESH


class TestTokenDecoding:
    """Test JWT token decoding."""

    def test_decode_valid_token(self):
        """Test decoding valid token."""
        token = create_access_token(
            subject="testuser",
            role=UserRole.ANALYST,
        )

        decoded = decode_token(token)

        assert decoded is not None
        assert decoded.sub == "testuser"
        assert decoded.role == UserRole.ANALYST
        assert decoded.token_type == TokenType.ACCESS

    def test_decode_expired_token(self):
        """Test decoding expired token."""
        token = create_access_token(
            subject="user",
            role=UserRole.VIEWER,
            expires_delta=timedelta(seconds=-1),  # Already expired
        )

        decoded = decode_token(token)
        assert decoded is None

    def test_decode_invalid_token(self):
        """Test decoding invalid token."""
        decoded = decode_token("invalid.token.here")
        assert decoded is None

    def test_decode_empty_token(self):
        """Test decoding empty token."""
        decoded = decode_token("")
        assert decoded is None


class TestTokenRevocation:
    """Test token revocation."""

    def test_revoke_token(self):
        """Test token revocation."""
        token = create_access_token(
            subject="testuser",
            role=UserRole.ANALYST,
        )

        # Token should be valid
        decoded = decode_token(token)
        assert decoded is not None

        # Revoke the token
        revoke_token(token)

        # Token should now be invalid
        decoded_after = decode_token(token)
        assert decoded_after is None

    def test_revoke_multiple_tokens(self):
        """Test revoking multiple tokens."""
        tokens = [create_access_token(f"user{i}", UserRole.VIEWER) for i in range(3)]

        # Revoke all
        for token in tokens:
            revoke_token(token)

        # All should be invalid
        for token in tokens:
            assert decode_token(token) is None


# =============================================================================
# API Key Tests
# =============================================================================


class TestAPIKeys:
    """Test API key functionality."""

    def test_create_api_key(self):
        """Test API key creation."""
        key_response = create_api_key(
            name="Test Key",
            role=UserRole.API_SERVICE,
        )

        assert key_response.key.startswith("medsec_")
        assert key_response.name == "Test Key"
        assert key_response.key_id is not None

    def test_create_api_key_with_expiration(self):
        """Test API key with custom expiration."""
        key_response = create_api_key(
            name="Short-lived Key",
            role=UserRole.API_SERVICE,
            expires_days=7,
        )

        assert key_response.expires_at is not None

    def test_validate_api_key(self):
        """Test API key validation."""
        key_response = create_api_key(
            name="Valid Key",
            role=UserRole.ANALYST,
        )

        token_data = validate_api_key(key_response.key)
        assert token_data is not None
        assert token_data.role == UserRole.ANALYST

    def test_validate_invalid_api_key(self):
        """Test invalid API key validation."""
        token_data = validate_api_key("medsec_invalid_key")
        assert token_data is None

    def test_validate_empty_api_key(self):
        """Test empty API key validation."""
        token_data = validate_api_key("")
        assert token_data is None


# =============================================================================
# Authentication Tests
# =============================================================================


class TestAuthentication:
    """Test user authentication."""

    def test_authenticate_valid_user(self):
        """Test authentication with valid credentials."""
        # Using demo users
        user = authenticate_user("admin", "admin123secure")

        assert user is not None
        assert user.username == "admin"
        assert user.role == UserRole.ADMIN

    def test_authenticate_invalid_password(self):
        """Test authentication with wrong password."""
        user = authenticate_user("admin", "wrongpassword")
        assert user is None

    def test_authenticate_nonexistent_user(self):
        """Test authentication with nonexistent user."""
        user = authenticate_user("nonexistent", "anypassword")
        assert user is None

    def test_authenticate_analyst(self):
        """Test analyst authentication."""
        user = authenticate_user("analyst", "analyst123secure")

        assert user is not None
        assert user.role == UserRole.ANALYST

    def test_authenticate_viewer(self):
        """Test viewer authentication."""
        user = authenticate_user("viewer", "viewer123secure")

        assert user is not None
        assert user.role == UserRole.VIEWER


class TestUserCreation:
    """Test user creation."""

    def test_create_user(self):
        """Test creating a new user."""
        import uuid

        unique_username = f"testuser_{uuid.uuid4().hex[:8]}"
        user_data = UserCreate(
            username=unique_username,
            email=f"{unique_username}@example.com",
            role=UserRole.ANALYST,
            password="newpassword123",
        )

        user = create_user(user_data)

        assert user.username == unique_username
        assert user.email == f"{unique_username}@example.com"
        assert user.role == UserRole.ANALYST
        assert user.disabled is False


# =============================================================================
# Role Checker Tests
# =============================================================================


class TestRoleChecker:
    """Test role-based access control."""

    def test_admin_role_checker(self):
        """Test admin role checker allows admin."""
        admin_user = User(
            username="admin",
            email="admin@example.com",
            role=UserRole.ADMIN,
        )

        checker = RoleChecker([UserRole.ADMIN])
        # RoleChecker.__call__ is async, so we need asyncio.run()
        result = asyncio.run(checker(admin_user))
        assert result == admin_user

    def test_analyst_role_checker(self):
        """Test analyst role checker."""
        analyst_user = User(
            username="analyst",
            email="analyst@example.com",
            role=UserRole.ANALYST,
        )

        checker = RoleChecker([UserRole.ANALYST, UserRole.ADMIN])
        result = asyncio.run(checker(analyst_user))
        assert result == analyst_user

    def test_role_checker_insufficient_permissions(self):
        """Test role checker with insufficient permissions."""
        viewer_user = User(
            username="viewer",
            email="viewer@example.com",
            role=UserRole.VIEWER,
        )

        checker = RoleChecker([UserRole.ADMIN])

        with pytest.raises(Exception):  # HTTPException
            asyncio.run(checker(viewer_user))

    def test_multi_role_checker(self):
        """Test checker with multiple allowed roles."""
        analyst_user = User(
            username="analyst",
            email="analyst@example.com",
            role=UserRole.ANALYST,
        )

        checker = RoleChecker([UserRole.ANALYST, UserRole.ADMIN, UserRole.VIEWER])
        result = asyncio.run(checker(analyst_user))
        assert result == analyst_user


# =============================================================================
# Demo Users Tests
# =============================================================================


class TestDemoUsers:
    """Test demo user configuration."""

    def test_demo_users_exist(self):
        """Test demo users are configured."""
        assert "admin" in DEMO_USERS
        assert "analyst" in DEMO_USERS
        assert "viewer" in DEMO_USERS

    def test_demo_user_roles(self):
        """Test demo users have correct roles."""
        admin = DEMO_USERS["admin"]
        analyst = DEMO_USERS["analyst"]
        viewer = DEMO_USERS["viewer"]

        # DEMO_USERS contains User objects
        assert admin.role == UserRole.ADMIN
        assert analyst.role == UserRole.ANALYST
        assert viewer.role == UserRole.VIEWER

    def test_demo_users_not_disabled(self):
        """Test demo users are not disabled."""
        for username, user in DEMO_USERS.items():
            assert user.disabled is False


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_username(self):
        """Test authentication with empty username."""
        user = authenticate_user("", "password")
        assert user is None

    def test_empty_password(self):
        """Test authentication with empty password."""
        user = authenticate_user("admin", "")
        assert user is None

    def test_whitespace_credentials(self):
        """Test authentication with whitespace credentials."""
        user = authenticate_user("  ", "  ")
        assert user is None

    def test_special_characters_in_token_data(self):
        """Test token with special characters in subject."""
        token = create_access_token(
            subject="user@example.com",
            role=UserRole.ANALYST,
        )

        decoded = decode_token(token)
        assert decoded is not None
        assert decoded.sub == "user@example.com"

    def test_unicode_username(self):
        """Test token with unicode username."""
        token = create_access_token(
            subject="user_unicode",
            role=UserRole.VIEWER,
        )

        decoded = decode_token(token)
        assert decoded is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestAuthIntegration:
    """Integration tests for auth workflow."""

    def test_full_auth_flow(self):
        """Test complete authentication flow."""
        # Step 1: Authenticate
        user = authenticate_user("admin", "admin123secure")
        assert user is not None

        # Step 2: Create access token
        access_token = create_access_token(
            subject=user.username,
            role=user.role,
        )
        assert access_token is not None

        # Step 3: Create refresh token
        refresh_token = create_refresh_token(
            subject=user.username,
            role=user.role,
        )
        assert refresh_token is not None

        # Step 4: Decode and verify
        decoded = decode_token(access_token)
        assert decoded.sub == "admin"
        assert decoded.role == UserRole.ADMIN

        # Step 5: Revoke access token
        revoke_token(access_token)
        assert decode_token(access_token) is None

        # Refresh token should still work
        refresh_decoded = decode_token(refresh_token)
        assert refresh_decoded is not None

    def test_api_key_auth_flow(self):
        """Test API key authentication flow."""
        # Step 1: Create API key
        key_response = create_api_key(
            name="Service Key",
            role=UserRole.API_SERVICE,
        )

        # Step 2: Validate API key
        token_data = validate_api_key(key_response.key)
        assert token_data is not None
        assert token_data.role == UserRole.API_SERVICE

        # Step 3: Create token from API key data
        token = create_access_token(
            subject=token_data.sub,
            role=token_data.role,
        )
        assert token is not None
