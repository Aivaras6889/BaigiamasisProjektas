import pandas as pd
import os
import logging
from pathlib import Path
import glob
from app.extensions import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiCSVConverter:
    def __init__(self, db):
        self.db = db
        self.class_names = {}

    def load_from_class_folders(self, base_directory, is_training=True, csv_filename="*.csv"):
        """
        Load data from folders where each class has its own CSV file
        
        Args:
            base_directory: Root directory containing class folders
            is_training: Boolean indicating if this is training data
            csv_filename: Pattern to match CSV files (default: "*.csv")
        """
        try:
            logger.info(f"Loading data from class folders in: {base_directory}")
            
            # Get all subdirectories (class folders)
            class_folders = [d for d in os.listdir(base_directory) 
                           if os.path.isdir(os.path.join(base_directory, d))]
            
            # Try to sort numerically if folders are numeric
            try:
                class_folders = sorted(class_folders, key=int)
            except ValueError:
                class_folders = sorted(class_folders)
            
            logger.info(f"Found {len(class_folders)} class folders: {class_folders}")
            
            total_imported = 0
            total_failed = 0
            
            for class_folder in class_folders:
                class_path = os.path.join(base_directory, class_folder)
                
                # Try to determine class ID
                try:
                    class_id = int(class_folder)
                except ValueError:
                    # If folder name is not numeric, use index
                    class_id = class_folders.index(class_folder)
                
                logger.info(f"Processing class {class_id} from folder: {class_folder}")
                
                # Find CSV files in this class folder
                csv_files = glob.glob(os.path.join(class_path, csv_filename))
                
                if not csv_files:
                    logger.warning(f"No CSV files found in {class_path}")
                    continue
                
                # Process each CSV file (usually just one per class)
                for csv_file in csv_files:
                    imported, failed = self._process_class_csv(
                        csv_file, class_path, class_id, is_training
                    )
                    total_imported += imported
                    total_failed += failed
            
            logger.info(f"Total import completed: {total_imported} successful, {total_failed} failed")
            return total_imported, total_failed
            
        except Exception as e:
            logger.error(f"Error loading from class folders: {str(e)}")
            raise
    
    def _process_class_csv(self, csv_file, class_directory, class_id, is_training):
        """Process a single CSV file for a specific class"""
        try:
            logger.info(f"Processing CSV: {csv_file}")
            
            # Read CSV file
            df = pd.read_csv(csv_file, sep=';')  # Many datasets use semicolon separator
            if df.empty:
                # Try comma separator if semicolon doesn't work
                df = pd.read_csv(csv_file, sep=',')
            
            logger.info(f"CSV columns: {list(df.columns)}")
            logger.info(f"Sample data:\n{df.head()}")
            
            imported_count = 0
            failed_count = 0
            
            for index, row in df.iterrows():
                try:
                    # Extract filename - try different column names
                    filename = self._extract_filename(row)
                    if not filename:
                        logger.warning(f"Could not extract filename from row {index}")
                        failed_count += 1
                        continue
                    
                    # Build full image path
                    image_path = os.path.join(class_directory, filename)
                    
                    # Check if image exists
                    if not os.path.exists(image_path):
                        # Try with different extensions
                        base_name = os.path.splitext(filename)[0]
                        extensions = ['.ppm', '.jpg', '.jpeg', '.png', '.bmp']
                        
                        found = False
                        for ext in extensions:
                            test_path = os.path.join(class_directory, base_name + ext)
                            if os.path.exists(test_path):
                                image_path = test_path
                                filename = base_name + ext
                                found = True
                                break
                        
                        if not found:
                            logger.warning(f"Image not found: {image_path}")
                            failed_count += 1
                            continue
                    
                    # Extract dimensions and ROI if available
                    width = self._safe_extract_numeric(row, ['Width', 'width', 'w'])
                    height = self._safe_extract_numeric(row, ['Height', 'height', 'h'])
                    
                    # Extract ROI coordinates
                    roi_coords = self._extract_roi_coordinates(row)
                    
                    # Add to database (no class_name needed)
                    record_id = self.db.add_traffic_sign(
                        filename=filename,
                        image_path=image_path,
                        class_id=class_id,
                        width=width,
                        height=height,
                        roi_coords=roi_coords,
                        is_training=is_training
                    )
                    
                    imported_count += 1
                    
                    if imported_count % 50 == 0:
                        logger.info(f"Processed {imported_count} images from class {class_id}")
                    
                except Exception as e:
                    logger.error(f"Error processing row {index} in {csv_file}: {str(e)}")
                    failed_count += 1
                    continue
            
            logger.info(f"Class {class_id} completed: {imported_count} imported, {failed_count} failed")
            return imported_count, failed_count
            
        except Exception as e:
            logger.error(f"Error processing CSV {csv_file}: {str(e)}")
            return 0, 1