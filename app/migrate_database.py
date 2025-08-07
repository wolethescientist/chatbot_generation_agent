#!/usr/bin/env python3
"""
Database migration script to add website_url column to chatbots table.
"""

import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.database import DATABASE_URL
from dotenv import load_dotenv

load_dotenv()

def migrate_database():
    """Add website_url column to chatbots table if it doesn't exist"""
    
    print("Starting database migration...")
    print(f"Database URL: {DATABASE_URL}")
    
    try:
        # Create engine and session
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        # Check if the column already exists
        print("Checking if website_url column exists...")
        
        check_column_query = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'chatbots' 
            AND column_name = 'website_url'
        """)
        
        result = session.execute(check_column_query)
        column_exists = result.fetchone() is not None
        
        if column_exists:
            print("✓ website_url column already exists. No migration needed.")
            return True
        
        # Add the website_url column
        print("Adding website_url column to chatbots table...")
        
        add_column_query = text("""
            ALTER TABLE chatbots 
            ADD COLUMN website_url VARCHAR NULL
        """)
        
        session.execute(add_column_query)
        session.commit()
        
        print("✓ Successfully added website_url column to chatbots table.")
        
        # Verify the column was added
        verify_result = session.execute(check_column_query)
        if verify_result.fetchone():
            print("✓ Migration verified successfully.")
            return True
        else:
            print("✗ Migration verification failed.")
            return False
            
    except Exception as e:
        print(f"✗ Migration failed: {str(e)}")
        if 'session' in locals():
            session.rollback()
        return False
        
    finally:
        if 'session' in locals():
            session.close()

def check_database_connection():
    """Test database connection"""
    print("Testing database connection...")
    
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            if result.fetchone():
                print("✓ Database connection successful.")
                return True
    except Exception as e:
        print(f"✗ Database connection failed: {str(e)}")
        return False
    
    return False

def main():
    """Run the migration"""
    print("=" * 50)
    print("Database Migration for Website URL Feature")
    print("=" * 50)
    
    # Check database connection first
    if not check_database_connection():
        print("\nPlease ensure your database is running and the connection details are correct.")
        print("Check your .env file for DATABASE_URL setting.")
        return False
    
    # Run migration
    success = migrate_database()
    
    if success:
        print("\n" + "=" * 50)
        print("Migration completed successfully!")
        print("You can now restart your FastAPI server.")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("Migration failed. Please check the error messages above.")
        print("=" * 50)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
