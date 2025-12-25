-- Row Level Security (RLS) Policies for Chainlytics
-- These policies ensure multi-tenancy and data isolation

-- Enable RLS on all tables
ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE warehouses ENABLE ROW LEVEL SECURITY;
ALTER TABLE clients ENABLE ROW LEVEL SECURITY;
ALTER TABLE stock ENABLE ROW LEVEL SECURITY;
ALTER TABLE vehicles ENABLE ROW LEVEL SECURITY;
ALTER TABLE priority_queue ENABLE ROW LEVEL SECURITY;
ALTER TABLE simulations ENABLE ROW LEVEL SECURITY;
ALTER TABLE simulation_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE policies ENABLE ROW LEVEL SECURITY;
ALTER TABLE metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Helper function to get the current user's organization ID
-- This assumes the JWT token contains the user's ID
CREATE OR REPLACE FUNCTION get_current_user_org_id()
RETURNS UUID AS $$
BEGIN
    -- This would be implemented based on how you pass the user context
    -- For example, from a JWT claim or session variable
    RETURN NULL; -- Placeholder
END;
$$ LANGUAGE plpgsql;

-- Organizations policies
-- Users can only see their own organization
CREATE POLICY org_select_policy ON organizations
    FOR SELECT
    USING (id = get_current_user_org_id());

-- Users table policies
-- Users can only see users from their organization
CREATE POLICY users_select_policy ON users
    FOR SELECT
    USING (organization_id = get_current_user_org_id());

CREATE POLICY users_insert_policy ON users
    FOR INSERT
    WITH CHECK (organization_id = get_current_user_org_id());

CREATE POLICY users_update_policy ON users
    FOR UPDATE
    USING (organization_id = get_current_user_org_id());

CREATE POLICY users_delete_policy ON users
    FOR DELETE
    USING (organization_id = get_current_user_org_id());

-- Warehouses policies
CREATE POLICY warehouses_select_policy ON warehouses
    FOR SELECT
    USING (organization_id = get_current_user_org_id());

CREATE POLICY warehouses_insert_policy ON warehouses
    FOR INSERT
    WITH CHECK (organization_id = get_current_user_org_id());

CREATE POLICY warehouses_update_policy ON warehouses
    FOR UPDATE
    USING (organization_id = get_current_user_org_id());

CREATE POLICY warehouses_delete_policy ON warehouses
    FOR DELETE
    USING (organization_id = get_current_user_org_id());

-- Clients policies
CREATE POLICY clients_select_policy ON clients
    FOR SELECT
    USING (organization_id = get_current_user_org_id());

CREATE POLICY clients_insert_policy ON clients
    FOR INSERT
    WITH CHECK (organization_id = get_current_user_org_id());

CREATE POLICY clients_update_policy ON clients
    FOR UPDATE
    USING (organization_id = get_current_user_org_id());

CREATE POLICY clients_delete_policy ON clients
    FOR DELETE
    USING (organization_id = get_current_user_org_id());

-- Stock policies
CREATE POLICY stock_select_policy ON stock
    FOR SELECT
    USING (organization_id = get_current_user_org_id());

CREATE POLICY stock_insert_policy ON stock
    FOR INSERT
    WITH CHECK (organization_id = get_current_user_org_id());

CREATE POLICY stock_update_policy ON stock
    FOR UPDATE
    USING (organization_id = get_current_user_org_id());

CREATE POLICY stock_delete_policy ON stock
    FOR DELETE
    USING (organization_id = get_current_user_org_id());

-- Vehicles policies
CREATE POLICY vehicles_select_policy ON vehicles
    FOR SELECT
    USING (organization_id = get_current_user_org_id());

CREATE POLICY vehicles_insert_policy ON vehicles
    FOR INSERT
    WITH CHECK (organization_id = get_current_user_org_id());

CREATE POLICY vehicles_update_policy ON vehicles
    FOR UPDATE
    USING (organization_id = get_current_user_org_id());

CREATE POLICY vehicles_delete_policy ON vehicles
    FOR DELETE
    USING (organization_id = get_current_user_org_id());

-- Priority queue policies
CREATE POLICY priority_queue_select_policy ON priority_queue
    FOR SELECT
    USING (organization_id = get_current_user_org_id());

CREATE POLICY priority_queue_insert_policy ON priority_queue
    FOR INSERT
    WITH CHECK (organization_id = get_current_user_org_id());

CREATE POLICY priority_queue_update_policy ON priority_queue
    FOR UPDATE
    USING (organization_id = get_current_user_org_id());

CREATE POLICY priority_queue_delete_policy ON priority_queue
    FOR DELETE
    USING (organization_id = get_current_user_org_id());

-- Simulations policies
CREATE POLICY simulations_select_policy ON simulations
    FOR SELECT
    USING (organization_id = get_current_user_org_id());

CREATE POLICY simulations_insert_policy ON simulations
    FOR INSERT
    WITH CHECK (organization_id = get_current_user_org_id());

CREATE POLICY simulations_update_policy ON simulations
    FOR UPDATE
    USING (organization_id = get_current_user_org_id());

CREATE POLICY simulations_delete_policy ON simulations
    FOR DELETE
    USING (organization_id = get_current_user_org_id());

-- Simulation logs policies
CREATE POLICY simulation_logs_select_policy ON simulation_logs
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM simulations 
            WHERE simulations.id = simulation_logs.simulation_id 
            AND simulations.organization_id = get_current_user_org_id()
        )
    );

-- Policies policies
CREATE POLICY policies_select_policy ON policies
    FOR SELECT
    USING (organization_id = get_current_user_org_id());

CREATE POLICY policies_insert_policy ON policies
    FOR INSERT
    WITH CHECK (organization_id = get_current_user_org_id());

CREATE POLICY policies_update_policy ON policies
    FOR UPDATE
    USING (organization_id = get_current_user_org_id());

CREATE POLICY policies_delete_policy ON policies
    FOR DELETE
    USING (organization_id = get_current_user_org_id());

-- Metrics policies
CREATE POLICY metrics_select_policy ON metrics
    FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM simulations 
            WHERE simulations.id = metrics.simulation_id 
            AND simulations.organization_id = get_current_user_org_id()
        )
    );

-- Audit logs policies
CREATE POLICY audit_logs_select_policy ON audit_logs
    FOR SELECT
    USING (organization_id = get_current_user_org_id());

CREATE POLICY audit_logs_insert_policy ON audit_logs
    FOR INSERT
    WITH CHECK (organization_id = get_current_user_org_id());

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO authenticated;