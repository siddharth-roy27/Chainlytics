# Chainlytics Architecture Summary

## Overview
This document provides a comprehensive overview of the Chainlytics architecture, focusing on the proper separation of authentication and authorization, and the enterprise-grade database schema.

## Authentication vs Authorization Model

### Key Principle
**Never duplicate authentication data.** Identity management is handled entirely by Nhost/Supabase.

### Architecture Layers

| Layer | Responsibility | Location | Managed By |
|-------|---------------|----------|------------|
| Identity | Who is the user? | `auth.users` | Nhost/Supabase |
| Authorization | What can the user do? | `public.users` | Chainlytics |

### Data Flow
1. User authenticates with Nhost (email/password or OAuth)
2. Nhost returns a JWT token containing the user's ID
3. Frontend includes JWT in API requests
4. Backend validates JWT and extracts user ID
5. Backend queries `public.users` for role/permission data
6. Backend enforces RBAC based on user role

## Database Schema Highlights

### Core Tables
1. **organizations** - Multi-tenant root
2. **users** - Authorization layer (references `auth.users.id`)
3. **warehouses** - Physical storage locations
4. **clients** - Customer entities
5. **stock** - Inventory management
6. **vehicles** - Transportation assets
7. **priority_queue** - Decision scheduling
8. **simulations** - Core ML objects
9. **policies** - RL/heuristic strategies
10. **audit_logs** - Security/compliance

### Key Design Principles
- **No auth duplication** - Prevents data inconsistency
- **Full auditability** - Every action is logged
- **Multi-tenant safe** - Data isolation between organizations
- **ML-ready** - Supports offline training and evaluation
- **Rollback-capable** - Historical data retention

## Security Implementation

### Row Level Security (RLS)
All tables implement RLS policies to ensure:
- Users can only access data from their organization
- Cross-tenant data leakage is impossible
- Fine-grained access control based on user roles

### JWT Verification
Backend API endpoints:
- Validate JWT signatures using Nhost's public keys
- Extract user IDs from verified tokens
- Reject expired or tampered tokens
- Map user IDs to authorization data

## Frontend Authentication Flow

### Current State
Using mock authentication for development

### Production Implementation
Will integrate with Nhost using:
- `@nhost/react` hooks for React components
- `@nhost/nhost-js` for client-side operations
- Proper JWT handling for API requests

## Backend Integration Points

### API Server Responsibilities
1. JWT verification for all protected endpoints
2. User authorization lookup from `public.users`
3. RBAC enforcement based on user roles
4. Data access restricted by organization boundaries
5. Audit logging for compliance

### Python Implementation Example
```python
@app.get("/api/warehouses")
def get_warehouses(request):
    # 1. Extract and verify JWT
    user_id = verify_jwt_get_user_id(request)
    
    # 2. Get user authorization
    auth_data = get_user_authorization(user_id)
    
    # 3. Query organization data only
    warehouses = query_warehouses(auth_data.org_id)
    
    return warehouses
```

## Deployment Considerations

### Environment Configuration
- Nhost project credentials (subdomain, region)
- JWT verification settings
- Database connection parameters
- RLS policy activation

### Scaling Strategy
- Connection pooling for database access
- Caching for frequently accessed authorization data
- Load balancing for API servers
- CDN for static assets

## Future Enhancements

### Short Term
1. Complete Nhost integration
2. Implement RLS policies
3. Add comprehensive audit logging
4. Set up monitoring and alerting

### Long Term
1. Advanced analytics dashboard
2. Machine learning model registry
3. Real-time collaboration features
4. Mobile application support

## Conclusion

The Chainlytics architecture follows enterprise best practices by:
- Separating authentication from authorization
- Implementing robust security measures
- Ensuring multi-tenancy and data isolation
- Supporting scalability and future growth
- Maintaining auditability and compliance

This foundation enables the development of a secure, scalable supply chain intelligence platform that can evolve with changing business requirements.