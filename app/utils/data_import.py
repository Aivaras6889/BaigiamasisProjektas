import os
import pandas as pd
import glob
from app.models.traffic_signs import TrafficSign

def import_class_folders(base_directory, db, is_training=True, csv_filename="*.csv"):
    class_folders = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    class_folders.sort(key=lambda x: int(x) if x.isdigit() else 999)

    total_imported = 0
    for class_folder in class_folders:
        class_id = int(class_folder)
        class_path = os.path.join(base_directory, class_folder)

        """ Find CSV files """
        csv_files = glob.glob(os.path.join(class_path, "*.csv"))
        if csv_files:
        
            csv_file=csv_files[0]
            try:
                df= pd.read_csv(csv_file, sep=';')
                if df.empty:
                    df = pd.read_csv(csv_file,sep=',')
            except:
                continue

            for _, row in df.iterrows():
                filename = None
                for col in ['Filename', 'filename','Path','path']:
                    if col in row and pd.notna(row[col]):
                        filename = str(row[col])
                        break

                if not filename:
                    continue

                # Build image path
                image_path = os.path.join(class_path, filename)
                if not os.path.exists(image_path):
                    base_name=os.path.splitext(filename)[0]
                    for ext in ['.ppm','.jpg', '.jpeg', '.png']:
                        test_path = os.path.join(class_path, base_name + ext)
                        if os.path.exists(test_path):
                            image_path = test_path
                            filename = base_name + ext
                            break
                if not os.path.exists(image_path):
                    continue

                # Extract metadata
                width = row.get('Width', 0)
                height = row.get('Height', 0)

                roi_coords = None
                if all(col in row for col in ['ROI_X1', 'ROI_Y1', 'ROI_X2', 'ROI_Y2']):
                    try:
                        roi_coords = (
                            int(row['ROI.X1']),
                            int(row['ROI.Y1']),
                            int(row['ROI.X2']),
                            int(row['ROI.Y2'])
                        )
                    except ValueError:
                        pass  # Ignore if conversion fails

                    sign = TrafficSign(
                        filename=filename,
                        image_path=image_path,
                        class_id=class_id,
                        width=int(width),
                        height=int(height),
                        roi_x1=roi_coords[0] if roi_coords else 0,
                        roi_y1=roi_coords[1] if roi_coords else 0,
                        roi_x2=roi_coords[2] if roi_coords else 0,
                        roi_y2=roi_coords[3] if roi_coords else 0,
                        is_training=is_training
                    )
                    db.session.add(sign)
                    total_imported += 1
        
            print(f"Class {class_id}: processed")
    db.session.commit()
    print(f"Total imported traffic signs: {total_imported}")
