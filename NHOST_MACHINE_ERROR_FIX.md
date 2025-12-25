# Nhost "Cannot read properties of undefined (reading 'machine')" Error Fix

## Issue
The application was throwing the error "Cannot read properties of undefined (reading 'machine')" when trying to run the development server. This error prevented the application from loading properly.

## Root Cause
The issue was caused by incorrect initialization of the Nhost client. The application was using `createNhostClient` from `@nhost/nhost-js` to initialize the client, but then trying to use it with `NhostProvider` from `@nhost/react`. This mismatch caused the error.

## Solution
Updated the Nhost client initialization in `/src/lib/nhost.ts` to use the correct method:

### Before (Incorrect)
```typescript
import { createNhostClient } from '@nhost/nhost-js';

export const nhost = createNhostClient({
  subdomain: import.meta.env.VITE_NHOST_SUBDOMAIN || 'zqxhlhebldmlgoegdhht',
  region: import.meta.env.VITE_NHOST_REGION || 'ap-south-1',
});
```

### After (Correct)
```typescript
import { createClient } from '@nhost/react';

export const nhost = createClient({
  subdomain: import.meta.env.VITE_NHOST_SUBDOMAIN || 'zqxhlhebldmlgoegdhht',
  region: import.meta.env.VITE_NHOST_REGION || 'ap-south-1',
});
```

## Explanation
The `@nhost/react` package provides its own client creation method (`createClient`) that is specifically designed to work with the `NhostProvider`. When using `NhostProvider` from `@nhost/react`, you should use the client created by `createClient` from the same package, not the one from `@nhost/nhost-js`.

## Verification
After making this change, the development server starts successfully without the error, and the application is accessible at http://localhost:8082.

## Related Documentation
According to Nhost documentation, when using React with Nhost:
1. Use `createClient` from `@nhost/react` for client initialization
2. Wrap your application with `NhostProvider` from `@nhost/react`
3. Use the same client instance for both initialization and provider

This ensures proper integration between the Nhost client and React provider, preventing internal errors like the "machine" property error.