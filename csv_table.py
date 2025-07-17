import csv

# write csv data to mysql database with sqlalchemy
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
def create_db_connection(db_url):
    """
    Create a database connection using SQLAlchemy.
    
    :param db_url: Database URL.
    :return: SQLAlchemy engine.
    """
    engine = create_engine(db_url)
    return engine

def create_table_from_csv(csv_file_path, table_name, db_connection):
    """
    Create a table in the database from a CSV file.
    
    :param csv_file_path: Path to the CSV file.
    :param table_name: Name of the table to create.
    :param db_connection: SQLAlchemy engine.
    """
    metadata = MetaData(bind=db_connection)
    table = Table(table_name, metadata, autoload_with=db_connection)

    with open(csv_file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        try:
            with db_connection.connect() as connection:
                for row in reader:
                    ins = table.insert().values(**row)
                    connection.execute(ins)
        except SQLAlchemyError as e:
            print(f"Error occurred: {e}")
    print(f"Table '{table_name}' created successfully from '{csv_file_path}'.")
# bulk csv data to mysql database with sqlalchemy
def bulk_insert_from_csv(csv_file_path, table_name, db_connection):
    """
    Bulk insert data from a CSV file into a database table.
    
    :param csv_file_path: Path to the CSV file.
    :param table_name: Name of the table to insert data into.
    :param db_connection: SQLAlchemy engine.
    """
    metadata = MetaData(bind=db_connection)
    table = Table(table_name, metadata, autoload_with=db_connection)

    with open(csv_file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        try:
            with db_connection.connect() as connection:
                connection.execute(table.insert(), [row for row in reader])
        except SQLAlchemyError as e:
            print(f"Error occurred during bulk insert: {e}")
    print(f"Bulk insert completed for table '{table_name}' from '{csv_file_path}'.")

    # how would look like function which reads dirs and subdirs if csv files and inserts them into database
def insert_csv_files_from_directory(directory_path, db_url):
    """
    Insert CSV files from a directory and its subdirectories into the database.
    :param directory_path: Path to the directory containing CSV files.
    :param db_url: Database URL.
    """
    import os
    engine = create_db_connection(db_url)
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                csv_file_path = os.path.join(root, file)
                table_name = os.path.splitext(file)[0]  # Use the file name as the table name
                create_table_from_csv(csv_file_path, table_name, engine)
                bulk_insert_from_csv(csv_file_path, table_name, engine)

#after uploading image asign image for example train test
def assign_image_to_dataset(image_path, dataset_name, db_connection):
    """
    Assign an image to a specific dataset in the database.
    
    :param image_path: Path to the image file.
    :param dataset_name: Name of the dataset to assign the image to.
    :param db_connection: SQLAlchemy engine.
    """
    metadata = MetaData(bind=db_connection)
    datasets_table = Table('datasets', metadata, autoload_with=db_connection)
    
    with db_connection.connect() as connection:
        ins = datasets_table.insert().values(image_path=image_path, dataset_name=dataset_name)
        connection.execute(ins)
    print(f"Image '{image_path}' assigned to dataset '{dataset_name}'.")