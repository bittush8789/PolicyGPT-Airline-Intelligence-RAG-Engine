-- PostgreSQL Initialization Script
-- Creates vectordb and statedb with required extensions

CREATE DATABASE vectordb;
CREATE DATABASE statedb;

\c vectordb;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

\c statedb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
