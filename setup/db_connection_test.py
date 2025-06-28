import psycopg2

try:
    conn = psycopg2.connect(
        database="vectordb",
        user="postgres",
        password="password",
        host="127.0.0.1",
        port=5432
    )
    print("✅ Connection successful!")
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")
