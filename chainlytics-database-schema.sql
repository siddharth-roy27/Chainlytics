-- Chainlytics Database Schema (PostgreSQL)
-- Enterprise-grade, multi-tenant, ML-ready

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1️⃣ Organizations (Multi-Tenant Root)
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    plan_type TEXT CHECK (plan_type IN ('free', 'pro', 'enterprise')) DEFAULT 'free',
    created_at TIMESTAMP DEFAULT now()
);

-- 2️⃣ Users (Authorization Layer ONLY)
-- NOTE: This table references auth.users.id from Nhost/Supabase
-- No email, password, or OAuth fields here - those are managed by the auth provider
CREATE TABLE users (
    id UUID PRIMARY KEY, -- MUST match auth.users.id from Nhost
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    role TEXT CHECK (role IN ('admin', 'analyst', 'viewer')) NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

-- 3️⃣ Warehouses
CREATE TABLE warehouses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    location JSONB,
    max_capacity INTEGER,
    priority INTEGER DEFAULT 0,
    status TEXT CHECK (status IN ('active', 'inactive')) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT now()
);

-- 4️⃣ Clients
CREATE TABLE clients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    location JSONB,
    default_order_size INTEGER,
    priority INTEGER DEFAULT 0,
    status TEXT CHECK (status IN ('active', 'inactive')) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT now()
);

-- 5️⃣ Stock / Inventory
CREATE TABLE stock (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    warehouse_id UUID REFERENCES warehouses(id) ON DELETE CASCADE,
    sku TEXT NOT NULL,
    product_name TEXT,
    quantity INTEGER NOT NULL,
    max_capacity INTEGER,
    alert_threshold INTEGER,
    stock_type TEXT CHECK (stock_type IN ('raw', 'finished')),
    status TEXT DEFAULT 'available',
    created_at TIMESTAMP DEFAULT now()
);

-- 6️⃣ Vehicles / Fleet
CREATE TABLE vehicles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    warehouse_id UUID REFERENCES warehouses(id),
    name TEXT,
    type TEXT,
    capacity INTEGER,
    priority INTEGER DEFAULT 0,
    status TEXT CHECK (status IN ('idle', 'in_transit', 'maintenance')),
    created_at TIMESTAMP DEFAULT now()
);

-- 7️⃣ Priority Queue (Decision Scheduling)
CREATE TABLE priority_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    entity_type TEXT CHECK (entity_type IN ('warehouse', 'client', 'vehicle', 'order')),
    entity_id UUID NOT NULL,
    priority_order INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

-- 8️⃣ Simulations (Core ML Object)
CREATE TABLE simulations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    created_by UUID REFERENCES users(id),
    status TEXT CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    timesteps INTEGER,
    params JSONB,
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT now()
);

-- 9️⃣ Simulation Logs (Explainability & Debug)
CREATE TABLE simulation_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    simulation_id UUID REFERENCES simulations(id) ON DELETE CASCADE,
    timestamp TIMESTAMP DEFAULT now(),
    message TEXT
);

-- 🔟 Policies (RL / Heuristic)
CREATE TABLE policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    name TEXT,
    policy_type TEXT CHECK (policy_type IN ('rl', 'heuristic')),
    config JSONB,
    model_version TEXT,
    status TEXT CHECK (status IN ('active', 'inactive')),
    created_at TIMESTAMP DEFAULT now()
);

-- 1️⃣1️⃣ Metrics (Training + Evaluation)
CREATE TABLE metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    simulation_id UUID REFERENCES simulations(id) ON DELETE CASCADE,
    warehouse_id UUID,
    client_id UUID,
    total_cost NUMERIC,
    avg_delay NUMERIC,
    stockouts INTEGER,
    created_at TIMESTAMP DEFAULT now()
);

-- 1️⃣2️⃣ Audit Logs (Security & Compliance)
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id),
    user_id UUID REFERENCES users(id),
    action TEXT,
    entity_type TEXT,
    entity_id UUID,
    timestamp TIMESTAMP DEFAULT now(),
    metadata JSONB
);

-- Indexes for performance
CREATE INDEX idx_users_organization_id ON users(organization_id);
CREATE INDEX idx_warehouses_organization_id ON warehouses(organization_id);
CREATE INDEX idx_clients_organization_id ON clients(organization_id);
CREATE INDEX idx_stock_organization_id ON stock(organization_id);
CREATE INDEX idx_vehicles_organization_id ON vehicles(organization_id);
CREATE INDEX idx_priority_queue_organization_id ON priority_queue(organization_id);
CREATE INDEX idx_simulations_organization_id ON simulations(organization_id);
CREATE INDEX idx_simulations_created_by ON simulations(created_by);
CREATE INDEX idx_simulation_logs_simulation_id ON simulation_logs(simulation_id);
CREATE INDEX idx_policies_organization_id ON policies(organization_id);
CREATE INDEX idx_metrics_simulation_id ON metrics(simulation_id);
CREATE INDEX idx_audit_logs_organization_id ON audit_logs(organization_id);
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);

-- Comments for documentation
COMMENT ON TABLE users IS 'Authorization layer only - references auth.users.id from Nhost. No email/password fields here.';
COMMENT ON COLUMN users.id IS 'MUST match auth.users.id from Nhost/Supabase authentication system';
COMMENT ON COLUMN simulations.created_by IS 'References users.id for audit trail';