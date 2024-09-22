import os
import torch

class PrototypeCombiner:
    def __init__(self):
        # Standardverzeichnisse und Dateinamen definieren
        self.parts_dir = os.path.join("prototypes", "parts")  # Verzeichnis mit den Originaldateien
        self.parts_target_dir = os.path.join(self.parts_dir, "all")  # Verzeichnis für die kombinierten Dateien
        self.tools_dir = os.path.join("prototypes", "tools")  # Verzeichnis mit Werkzeug-Prototypen
        self.tools_output_file = "ycb_prototypes.pth"  # Name der kombinierten Werkzeuge-Datei

    def combine_pth_files(self, pth_files):
        """
        Kombiniert .pth-Dateien durch Konkatenieren ihrer 'prototypes' und Zusammenführen ihrer 'label_names'.
        """
        merged_dict = None

        for pth_file in pth_files:
            data = torch.load(pth_file)
            
            # Sicherstellen, dass 'label_names' eine Liste ist
            if isinstance(data['label_names'], str):
                data['label_names'] = [data['label_names']]
            
            if merged_dict is None:
                merged_dict = data
            else:
                merged_dict['prototypes'] = torch.cat((merged_dict['prototypes'], data['prototypes']), dim=0)
                merged_dict['label_names'] += data['label_names']
        
        return merged_dict

    def combine_part_prototypes(self):
        """
        Kombiniert Teil-Prototypen aus dem 'parts' Verzeichnis in einzelne .pth-Dateien pro Teiltyp.
        """
        # "all"-Verzeichnis erstellen, falls es noch nicht existiert
        os.makedirs(self.parts_target_dir, exist_ok=True)

        # Liste der gültigen Objekte (Verzeichnisse) im 'parts' Verzeichnis, außer 'all'
        valid_objects = {name for name in os.listdir(self.parts_dir) if os.path.isdir(os.path.join(self.parts_dir, name)) and name != "all"}

        for root, dirs, files in os.walk(self.parts_dir):
            if not files:
                continue

            # Nur .pth-Dateien filtern
            pth_files = [os.path.join(root, f) for f in files if f.endswith('.pth')]

            if pth_files:
                # Bestimmen des relativen Pfades des aktuellen Verzeichnisses in Bezug auf parts_dir
                relative_dir = os.path.relpath(root, self.parts_dir)
                relative_dir_parts = relative_dir.split(os.sep)

                # Der letzte Teil der Verzeichnisstruktur bestimmt den Namen der kombinierten .pth-Datei
                target_subdir_name = relative_dir_parts[0]
                if len(relative_dir_parts) > 1:
                    subfolder_name = relative_dir_parts[1]
                    combined_pth_filename = f"{target_subdir_name}_{subfolder_name}.pth"
                else:
                    combined_pth_filename = f"{target_subdir_name}.pth"

                # Prüfen, ob das Objekt im 'parts' Verzeichnis existiert (außer 'all')
                if target_subdir_name in valid_objects:
                    # Sicherstellen, dass das Zielverzeichnis existiert
                    target_subdir_path = os.path.join(self.parts_target_dir, target_subdir_name)
                    os.makedirs(target_subdir_path, exist_ok=True)

                    # Kombinieren der .pth-Dateien und Speichern des Ergebnisses
                    merged_data = self.combine_pth_files(pth_files)
                    target_pth_file = os.path.join(target_subdir_path, combined_pth_filename)
                    torch.save(merged_data, target_pth_file)
                    print(f"Gespeicherte kombinierte Datei unter {target_pth_file}")
                else:
                    print(f"Überspringe {target_subdir_name}, da es nicht im Verzeichnis parts existiert oder 'all' ist.")

    def combine_tool_prototypes(self):
        """
        Kombiniert Werkzeug-Prototypen aus dem 'tools' Verzeichnis in eine einzige .pth-Datei.
        """
        prototypes_list = []
        label_names_list = []

        for root, dirs, files in os.walk(self.tools_dir):
            for file in files:
                if file.endswith('.pth'):
                    data = torch.load(os.path.join(root, file))
                    prototypes = data['prototypes']
                    label_names = data['label_names']

                    prototypes_list.append(prototypes)
                    label_names_list.append(label_names)

        combined_data = {
            'prototypes': torch.cat(prototypes_list, dim=0),
            'label_names': label_names_list
        }

        torch.save(combined_data, self.tools_output_file)
        print(f"Gespeicherte kombinierte Werkzeug-Prototypen in {self.tools_output_file}")