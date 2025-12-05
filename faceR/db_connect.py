from pymongo import MongoClient

# Your connection string here
MONGO_URI = "mongodb+srv://ajaykathar30:EEk9w6fsnyYYLDGo@cluster0.441o7q2.mongodb.net/"

def get_database():
    try:
        client = MongoClient(MONGO_URI)
        db = client["medical_db"]   # You can name your DB here
        print("✅ MongoDB Connected Successfully!")
        return db
    except Exception as e:
        print("❌ Connection Error:", e)

if __name__ == "__main__":
    db = get_database()
    # Just to test
    print("Current DB Name:", db.name)
