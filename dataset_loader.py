import matrix_generator as mg
import y_dataframe_loader as ydf
import pandas as pd
import subprocess
from pathlib import Path


class FinalDatasetLoader:
    def __init__(self, base_path="~/graduate-paper-code/spe1"):
        self.base_path = Path(base_path).expanduser()
        self.source_file = self.base_path / "SPE1CASE1.data"
        self.generated_df = pd.DataFrame()

    def run_matrix_generator(self, iters):
        generator = mg.MatrixGenerator(
            iters=iters, 
            src_file='/home/dmitrysavinskikh/data/opm-data/spe1/SPE1CASE1.DATA', 
            main_dir='/home/dmitrysavinskikh/graduate-paper-code/spe1',
            swof_from=144,
            swof_to=157,
            sgof_from=169,
            sgof_to=182,
            pvdg_from=211,
            pvdg_to=219,
            pvto_from=229,
            pvto_to=235,
            equil_line=273
        )

        self.generated_df = generator.make_files()

        return self.generated_df
    
    def run_flow_on_all_files(
            self, 
            pattern="SPE1CASE1_ITER_*.DATA"
        ):        
        data_files = list(self.base_path.glob(pattern))
        
        if not data_files:
            print('empty: ', self.base_path)
        
        print('.DATA files ', len(data_files))
        
        self.flow_results = []
        for i, data_file in enumerate(data_files, 1):
            print(f"\n[{i}/{len(data_files)}] Processing: {data_file.name}")
            
            result = self._run_flow_command(data_file)
            self.flow_results.append({
                'file': data_file,
                'success': result['success'],
                'output': result['output'],
                'error': result['error']
            })
            
            if result['success']:
                print(f"  ✓ Successful end")
            else:
                print(f"  ✗ Flow with error")
        
        successful = sum(1 for r in self.flow_results if r['success'])
        print('Results: ')
        print('Amount: ', len(data_files))
        print('Successful: ', successful)
        print('With errors: ', len(data_files) - successful)
        
        return self.flow_results

    def _run_flow_command(self, data_file):
        """
        Run flow command for a specific file
        """
        command = f"flow {data_file}"
        
        try:
            # Запускаем процесс
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.base_path 
            )
            
            stdout, stderr = process.communicate(timeout=300)  # wait for the end, five minute timeout
            
            return {
                'success': process.returncode == 0,
                'output': stdout,
                'error': stderr,
                'returncode': process.returncode
            }
            
        except subprocess.TimeoutExpired:
            process.kill()
            return {
                'success': False,
                'output': '',
                'error': 'Timeout expired (300 seconds)',
                'returncode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': str(e),
                'returncode': -1
            }
        
    def get_successful_runs(self):
        return [r['file'] for r in self.flow_results if r['success']]
    
    def get_failed_runs(self):
        """
        Retrieves the list of files for which flow failed with an error
        """
        return [r['file'] for r in self.flow_results if not r['success']]
    
    def run_y_df(self):
        yloader = ydf.YDataFrameLoader()
        return yloader.load_final_df()

    def aggregate_dfs(self):
        x_df = self.run_matrix_generator(iters=3)
        y_df = self.run_y_df()

        return x_df, y_df


# if __name__ == "__main__":
#     loader = FinalDatasetLoader()
#     x_df = loader.run_matrix_generator(iters=1)[['iteration', 'N_w', 'N_o', 'coef_oil', 'coef_water']].drop_duplicates()
#     results = loader.run_flow_on_all_files()
#     y_df = loader.run_y_df()
#     y_df = y_df[y_df['day'] == 3650]
#     to_saving_df = pd.merge(x_df, y_df, on='iteration', how='inner')
#     print(to_saving_df)
#     yloader = ydf.YDataFrameLoader()
#     yloader.save_to_csv(df=to_saving_df,filename='csv_output.csv')
#     print('Successful runs: ', len(loader.get_successful_runs()))
#     print('Unsuccessful runs: ', len(loader.get_failed_runs()))

if __name__ == "__main__":
    loader = FinalDatasetLoader()
    x_df = loader.run_matrix_generator(iters=1)[['iteration', 'water_relative_permeability', 'oil_relative_permeability']].drop_duplicates()
    results = loader.run_flow_on_all_files()
    y_df = loader.run_y_df()
    y_df = y_df[y_df['day'] == 3650]
    to_saving_df = pd.merge(x_df, y_df, on='iteration', how='inner')
    print(to_saving_df)
    yloader = ydf.YDataFrameLoader()
    yloader.save_to_csv(df=to_saving_df,filename='csv_output.csv')
    print('Successful runs: ', len(loader.get_successful_runs()))
    print('Unsuccessful runs: ', len(loader.get_failed_runs()))