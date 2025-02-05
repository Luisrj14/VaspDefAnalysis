# Pakages
import os
import pathlib

class SaveOutput:
    """
    SaveOutput Class
    ----------------
    
    This class manages the saving of output figures (.png) and data files (.txt or .csv) 
    in an organized directory structure.
    """
    
    def __init__(self, directory_name_save_output: str) -> None:
        
        # Ensure the output name is a string
        if isinstance(directory_name_save_output, str):
            self.directory_name_save_output = directory_name_save_output
            self.base_path = pathlib.Path().absolute()  # Current working directory
            self.project_root_dir = os.path.join(self.base_path, self.directory_name_save_output)
            
            # Create root directory for saving if it doesn't exist
            if not os.path.exists(self.project_root_dir):
                os.makedirs(self.project_root_dir)
        else:
            raise TypeError(f"'{directory_name_save_output}' must be a string.")
    
    def figure_path(self,Figure_name: str=None,dir_name: str=None) -> str:
        """
        Generates the full path to save the figure, creates necessary directories if not already present.
        """
        if dir_name == None :
            figure_dir = os.path.join(self.project_root_dir, "FigureFiles")
        else:
             # Ensure the output dir_name is a string 
            if isinstance(dir_name, str):
                figure_dir = os.path.join(self.project_root_dir, dir_name)
            else: 
                raise TypeError(f"'{dir_name}' must be a string.")

        # Create figure directory if it doesn't exist
        os.makedirs(figure_dir, exist_ok=True)

        # Ensure the output Figure_name is a string 
        if isinstance(Figure_name, str):
            Figure_path = os.path.join(figure_dir, Figure_name)
        else: 
            raise TypeError(f"'{Figure_name}' must be a string.")

        return Figure_path
    
    def data_path(self,data_name: str= None, dir_name: str=None) -> str:
        """
        Generates the full path to save the data file, creates necessary directories if not already present.
        If the file doesn't exist, it will be created.
        """

        if dir_name == None :
            data_dir = os.path.join(self.project_root_dir, "DataFiles")
        else:
             # Ensure the output dir_name is a string 
            if isinstance(dir_name, str):
                data_dir = os.path.join(self.project_root_dir, dir_name)
            else: 
                raise TypeError(f"'{dir_name}' must be a string.")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Ensure the output Figure_name is a string 
        if isinstance(data_name, str):
            Data_file_path = os.path.join(data_dir, data_name)
        else: 
            raise TypeError(f"'{data_name}' must be a string.")

        
        # Create the data file if it doesn't exist
        if not os.path.exists(Data_file_path):
            creat_text = open(Data_file_path, 'w')
            creat_text.close()
        return Data_file_path

if __name__ == "__main__":
    # Example usage
    save_output = SaveOutput('output_folder')
    print("Figure path:", save_output.figure_path('example_figure.png'))
    print("Data path:", save_output.data_path('example_data.txt'))
