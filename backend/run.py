from app import create_app

app = create_app()

if __name__ == '__main__':
    # Force creation of database and other setup
    app.run(debug=True, port=5001)
