# Supabase Integration Guide for Chainlytics

## Overview
This guide explains how to properly integrate Chainlytics with Supabase for authentication while maintaining our separate authorization schema.

## Architecture

### Authentication vs Authorization
- **Authentication** (Identity): Handled by Supabase in `auth.users`
- **Authorization** (Permissions): Handled by us in `public.users`

### Data Flow
1. User authenticates with Supabase (email/password or OAuth)
2. Supabase returns a JWT token
3. Frontend uses JWT to access our API
4. Backend verifies JWT and extracts user ID
5. Backend queries our `public.users` table for authorization data

## Frontend Implementation

### 1. Install Dependencies
```bash
npm install @supabase/supabase-js
```

### 2. Create Supabase Client
```typescript
// src/lib/supabase.ts
import { createClient } from '@supabase/supabase-js';

export const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL,
  import.meta.env.VITE_SUPABASE_ANON_KEY
);
```

### 3. Update App.tsx
```typescript
// src/App.tsx
import { supabase } from '@/lib/supabase';

const App = () => (
  <div>
    {/* Rest of your app */}
  </div>
);
```

### 4. Update Authentication Logic
```typescript
// Authentication handled directly with supabase.auth
import { supabase } from '@/lib/supabase';

const handleLogin = async (email: string, password: string) => {
  const { error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });
  // Handle response
};

export const useAuth = () => {
  const { isAuthenticated, isLoading } = useAuthenticationStatus();
  const user = useUserData();
  const { signOut } = useSignOut();
  const { signInEmailPassword } = useSignInEmailPassword();

  const login = async (email: string, password: string) => {
    const { isError, error } = await signInEmailPassword(email, password);
    return { success: !isError, error };
  };

  const logout = async () => {
    await signOut();
  };

  return {
    user: user ? {
      id: user.id,
      email: user.email,
      displayName: user.displayName,
      // Map Nhost user fields to our user interface
    } : null,
    isAuthenticated,
    isLoading,
    login,
    logout
  };
};
```

## Backend Implementation

### 1. JWT Verification
```python
# api_server.py
import jwt
from cryptography.hazmat.primitives import serialization

def verify_nhost_jwt(token):
    """
    Verify JWT token issued by Nhost
    """
    try:
        # Get public key from Nhost JWKS endpoint
        # https://<subdomain>.<region>.nhost.run/v1/auth/jwks
        
        decoded = jwt.decode(
            token,
            public_key,  # Nhost's public key
            algorithms=['RS256'],
            audience='authenticated',
            issuer=f'https://{SUBDOMAIN}.{REGION}.nhost.run'
        )
        
        return decoded
    except jwt.InvalidTokenError as e:
        raise Exception(f"Invalid token: {str(e)}")
```

### 2. User Authorization Lookup
```python
# api_server.py
def get_user_authorization(nhost_user_id):
    """
    Get user authorization data from our database
    """
    # Query our users table using the Nhost user ID
    query = """
        SELECT u.role, o.id as organization_id, o.name as organization_name
        FROM users u
        JOIN organizations o ON u.organization_id = o.id
        WHERE u.id = %s
    """
    
    result = execute_query(query, (nhost_user_id,))
    if result:
        return result[0]
    return None
```

### 3. Protected Endpoint Example
```python
# api_server.py
@app.get("/api/warehouses")
async def get_warehouses(request: Request):
    # Extract JWT from Authorization header
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    
    token = auth_header.split(" ")[1]
    
    # Verify JWT
    try:
        nhost_user = verify_nhost_jwt(token)
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))
    
    # Get user authorization
    user_auth = get_user_authorization(nhost_user['sub'])
    if not user_auth:
        raise HTTPException(status_code=403, detail="User not authorized")
    
    # Query warehouses for user's organization
    warehouses = get_warehouses_for_organization(user_auth['organization_id'])
    
    return {"warehouses": warehouses}
```

## Database Schema Mapping

### Nhost auth.users Table (Managed by Nhost)
```
id (UUID) - Primary key
email (TEXT)
password (TEXT) - Hashed
email_verified (BOOLEAN)
disabled (BOOLEAN)
created_at (TIMESTAMP)
```

### Our public.users Table (Managed by Us)
```
id (UUID) - References auth.users.id
organization_id (UUID) - References organizations.id
role (TEXT) - 'admin', 'analyst', 'viewer'
created_at (TIMESTAMP)
```

## User Provisioning

### 1. Automatic Provisioning
When a user authenticates for the first time:

```python
def provision_user(nhost_user_id, email):
    """
    Automatically provision user in our authorization system
    """
    # Check if user already exists
    query = "SELECT id FROM users WHERE id = %s"
    result = execute_query(query, (nhost_user_id,))
    
    if not result:
        # Create default organization for user
        org_id = create_default_organization(email)
        
        # Insert user with default role
        insert_query = """
            INSERT INTO users (id, organization_id, role)
            VALUES (%s, %s, 'viewer')
        """
        execute_query(insert_query, (nhost_user_id, org_id))
```

### 2. Manual Provisioning
Admin users can manually provision users:

```sql
INSERT INTO users (id, organization_id, role)
VALUES ('<nhost_user_id>', '<org_id>', 'analyst');
```

## Security Best Practices

### 1. Never Store Passwords
Our database should never contain password hashes or authentication secrets.

### 2. Always Verify JWTs
Every API request should verify the JWT before processing.

### 3. Use Parameterized Queries
Prevent SQL injection by always using parameterized queries.

### 4. Implement Rate Limiting
Protect authentication endpoints from brute force attacks.

### 5. Monitor Suspicious Activity
Log authentication attempts and monitor for suspicious patterns.

## Testing

### 1. Unit Tests
Test JWT verification and user authorization lookup functions.

### 2. Integration Tests
Test complete authentication flow with Nhost staging environment.

### 3. Security Tests
Validate that users cannot access data from other organizations.

## Troubleshooting

### Common Issues

1. **JWT Verification Fails**
   - Check that the public key matches Nhost's current key
   - Verify the issuer URL matches your Nhost project

2. **User Not Found in Authorization Table**
   - Ensure automatic provisioning is working
   - Check that the Nhost user ID matches our user ID

3. **Permission Denied Errors**
   - Verify RLS policies are correctly implemented
   - Check that the user has the correct role for the operation

## Next Steps

1. Implement JWT verification in the backend
2. Set up automatic user provisioning
3. Configure RLS policies
4. Test the complete authentication flow
5. Deploy to staging environment for validation
