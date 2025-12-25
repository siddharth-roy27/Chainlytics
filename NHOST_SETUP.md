# Supabase Setup Guide for Chainlytics

## Overview
This guide explains how to set up Supabase for authentication and database services in the Chainlytics application.

## Prerequisites
- Node.js and npm installed
- Supabase account (https://supabase.com)

## Setting up Supabase

### 1. Install Supabase CLI
```bash
# Using npm
npm install -g supabase

# Or using other methods
curl -fsSL https://get.supabase.com | sh
```

### 2. Create a New Supabase Project
1. Log in to your Supabase account
2. Create a new project in the Supabase dashboard
3. Note down your:
   - Project URL
   - Project API Key (anon/public key)
   - Database connection string

### 3. Configure Environment Variables
Update your `.env` file with the Supabase configuration:

```env
# Supabase Configuration
VITE_SUPABASE_URL=https://your-project-id.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key-here
```

### 4. Set up Authentication
The frontend is already configured to use Supabase authentication. You need to:

1. Enable Email/Password authentication in the Supabase dashboard (Authentication → Settings)
2. Enable Google OAuth if needed:
   - Go to Authentication → Providers → Google
   - Add your Google Client ID and Client Secret
   - Set redirect URL: `https://your-project-id.supabase.co/auth/v1/callback`

### 5. Set up Database
Supabase provides a PostgreSQL database. To set it up:

1. Go to the "SQL Editor" section in your Supabase dashboard
2. Run the provided SQL schema to create tables
3. For Chainlytics, tables will be created for:
   - Users
   - Warehouses
   - Vehicles
   - Orders
   - Clients
   - Stock inventory
   - Simulation results

### 6. Connect to the Database
The frontend connects to Supabase using the REST API and Realtime features. You can also connect directly using the Supabase dashboard:

1. Go to the "Table Editor" section in your Supabase dashboard
2. Use the SQL Editor to run queries
3. Set up Row Level Security (RLS) policies for different user roles

## Testing the Setup

### 1. Test Authentication
Visit the login/signup pages in your frontend to test the authentication functionality.

### 2. Test Database Connection
You can test the database connection by making API calls through the frontend or using the Supabase dashboard.

## Troubleshooting

### Common Issues
1. **Environment variables not loading**: Make sure you're using the `VITE_` prefix for client-side environment variables
2. **Authentication failing**: Check that the Supabase project settings match your environment variables
3. **Database connection issues**: Verify the Supabase URL and API key

### Getting Help
- Check the [Supabase Documentation](https://supabase.com/docs)
- Visit the [Supabase Community](https://github.com/supabase/supabase/discussions)
- Contact Supabase support through your dashboard
