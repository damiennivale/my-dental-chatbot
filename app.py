if __name__ == "__main__":
    print("=" * 60)
    print("🦷 MyDental Chatbot")
    print("=" * 60)
    print("📍 http://localhost:5000")
    print("=" * 60)

    import webbrowser
    import threading
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5000")).start()

    # debug=False and 127.0.0.1: the Werkzeug debugger on an open host
    # lets anyone on your network run code on your machine.
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)