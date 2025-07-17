from extensions import db
import csv


def csv_to_db(file_path, model):
    """Import data from a CSV file into the database."""
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            instance = model(**row)
            db.session.add(instance)
        db.session.commit()

def csv_to_db_bulk(file_path, model):
    """Bulk import data from a CSV file into the database."""
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        instances = [model(**row) for row in reader]
        db.session.bulk_save_objects(instances)
        db.session.commit()

def read_csv_from_directory(directory_path):
    """Read all CSV files from a directory and return their paths."""
    import os
    csv_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def csvs_to_db_from_directory(directory_path, model):
    """Import all CSV files from a directory into the database."""
    csv_files = read_csv_from_directory(directory_path)
    for csv_file in csv_files:
        csv_to_db(csv_file, model)
        print(f"Imported {csv_file} into the database.")
