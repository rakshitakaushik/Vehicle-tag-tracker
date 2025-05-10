import pymysql
print("Before connect")

try:
    conn = pymysql.connect(
        host='127.0.0.1',
        user='rakshita2002',
        password='HelloMYSQL2002',
        database='vehicle_tracker',
        port=3307,
        connect_timeout=5
    )
    print("✅ Connected successfully.")
except Exception as e:
    print("❌ Connection failed:")
    import traceback
    traceback.print_exc()




