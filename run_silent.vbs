Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\Bryan\OneDrive\Documents\DENTEX\dental-ai-screening"
WshShell.Run "cmd /c "".venv\Scripts\streamlit.exe run app.py --server.headless true --browser.gatherUsageStats false""", 0, False
