import mysql.connector

# Database Connection Function
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="fdsr_db"
    )

# Function to insert new decision methods
def insert_decision_method(method_name, description, source):
    db = connect_db()
    cursor = db.cursor()

    sql = "INSERT INTO sample_decision_methods (method_name, description, source) VALUES (%s, %s, %s)"
    values = (method_name, description, source)

    cursor.execute(sql, values)
    db.commit()
    cursor.close()
    db.close()

    print(f"✅ Inserted: {method_name}")

# Function to search decision methods dynamically
def search_decision_methods(method=None, criteria=None, source=None, category=None):
    db = connect_db()
    cursor = db.cursor(dictionary=True)

    query = "SELECT * FROM sample_decision_methods WHERE 1=1"
    values = []

    if method:
        query += " AND LOWER(method_name) LIKE LOWER(%s)"
        values.append(f"%{method}%")

    if category:
        query += " OR LOWER(category) LIKE LOWER(%s)"
        values.append(f"%{category}%")

    if criteria:
        query += " OR LOWER(description) LIKE LOWER(%s)"
        values.append(f"%{criteria}%")

    if source:
        query += " OR LOWER(source) LIKE LOWER(%s)"
        values.append(f"%{source}%")

    cursor.execute(query, values)
    results = cursor.fetchall()
    cursor.close()
    db.close()

    return results
