import sqlite3
import pandas as pd

def init_db():
    conn = sqlite3.connect('medical_records.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            age INTEGER,
            sex TEXT,
            weight REAL,
            height REAL,
            occupation TEXT,
            prediction TEXT,
            severity TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(name, age, sex, weight, height, occupation, pred, sev, conf):
    conn = sqlite3.connect('medical_records.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO scans (
            patient_name, age, sex, weight, height, 
            occupation, prediction, severity, confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (name, age, sex, weight, height, occupation, pred, sev, conf))
    conn.commit()
    conn.close()