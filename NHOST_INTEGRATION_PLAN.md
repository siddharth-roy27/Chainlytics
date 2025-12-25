# Nhost Integration Plan for Chainlytics

## Current State Analysis
The application currently uses a mock authentication system that accepts any credentials, which is not suitable for production.

## Required Changes

### 1. Update Frontend Authentication Context
Replace the mock AuthContext with real Nhost integration:

1. Install Nhost React SDK:
   ```bash
   npm install @nhost/react @nhost/nhost-js
   ```

2. Create proper Nhost client configuration
3. Update AuthContext to use real Nhost authentication
4. Modify all components to use Nhost hooks properly

### 2. Backend Integration Points
The backend API needs to:
1. Verify JWT tokens from Nhost
2. Extract user information from verified tokens
3. Map Nhost user IDs to our authorization schema

### 3. Database Schema Alignment
Our schema correctly separates:
- `auth.users` (managed by Nhost - identity)
- `public.users` (our table - authorization)

We reference `auth.users.id` in our `public.users.id` column.

## Implementation Steps

### Phase 1: Frontend Nhost Integration
1. Install Nhost packages
2. Create Nhost client configuration
3. Replace AuthProvider with NhostProvider
4. Update all authentication hooks
5. Test login/logout flows

### Phase 2: Backend JWT Verification
1. Implement JWT verification middleware
2. Extract user ID from tokens
3. Query our authorization tables
4. Enforce RBAC based on user roles

### Phase 3: Data Synchronization
1. Set up user provisioning in our tables
2. Map Nhost user IDs to organizations and roles
3. Handle user deletions/updates

## Security Considerations
1. Never store passwords in our database
2. Always verify JWT tokens before processing requests
3. Implement proper row-level security
4. Use parameterized queries to prevent injection
5. Encrypt sensitive data in transit and at rest

## Testing Strategy
1. Unit tests for authentication flows
2. Integration tests with Nhost staging environment
3. End-to-end tests for critical user journeys
4. Security scanning for vulnerabilities