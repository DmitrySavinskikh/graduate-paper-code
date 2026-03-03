import re
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Union

class YDataFrameLoader:
    """
    Loader of data from SPE1CASE<N> PRT files
    Parses the files and creates a final dataframe with the simulation results.
    """
    
    def __init__(self, base_path: str = '~/graduate-paper-code/spe1'):
        self.base_path = Path(base_path).expanduser()
        self.parsed_data: Dict[str, pd.DataFrame] = {}
        
    def load_final_df(self, pattern: str = 'SPE1CASE1_ITER_*.PRT') -> pd.DataFrame:
        """
        Downloads and parses all .PRT files matching the pattern
        
        Parameters:
        pattern: pattern for search .PRT files
        
        Returns:
        Whole DataFrame with all data
        """
        print('Download from ', self.base_path)
        
        prt_files = list(self.base_path.glob(pattern))        
        print(f'amount .PRT files: {len(prt_files)}')
        
        all_dfs = []
        for prt_file in prt_files:
            print(f'processing for: {prt_file.name}')

            df = self.parse_prt_file(prt_file)
            if df is not None and not df.empty:
                iteration = self._extract_iteration_number(prt_file)
                df['iteration'] = iteration
                
                all_dfs.append(df)
                print(f'loaded {len(df)} rows')
                self.parsed_data[prt_file.name] = df
            else: 
                print('error with parsing .DATA file')
        
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            print('--------------------------------')
            print('FINAL DF: ')
            print(' - all rows: ',len(final_df))
            print(' - columns list: ',list(final_df.columns))
            print(' - unique iters: ', final_df['iteration'].nunique())
            print('--------------------------------')
            
            return final_df
        else:
            print('EMPTY DF!!!')
            return pd.DataFrame()
    
    def parse_prt_file(self, file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Parses a single .PRT and extract data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print('not found: ', file_path)
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print('error when I read this file: ', file_path)
            print(e)
            return None
        
        lines = content.split('\n')
        records = []
        current_day = None
        mark_line = None
        for _, line in enumerate(lines):
            # to define which day is it, we can try to find '  DAYS' in line
            # and count 8 (for example) elems before the first index, it will be enough
            if line.find('  DAYS') != -1:
                mark_line = None
                day_str = line[line.find('  DAYS') - 8:line.find('  DAYS')].strip()
                try:
                    current_day = int(day_str)
                except ValueError:
                    continue
            if 'CUMULATIVE PRODUCTION/INJECTION TOTALS' in line: # this part we need to parse too
                mark_line = line
            if mark_line is not None and (line.startswith(':PROD') or line.startswith(':INJ')): # only injection or production data

                record = self._parse_well_line(line, current_day)
                if record:
                    records.append(record)
        if not records:
            return None
        
        df = pd.DataFrame(records)
        # create correct types
        numeric_cols = ['day', 'oil_prod', 'water_prod', 'gas_prod', 
                       'oil_inj', 'water_inj', 'gas_inj', 'res_vol']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def _parse_well_line(self, line: str, day: float) -> Optional[Dict]:
        """
        Parse string with data for a day
        """
        parts = [p.strip() for p in line.split(':') if p.strip()]
                
        # if parts[0] in ('INJ', 'PROD'):
        #     return {
        #         'day': day,
        #         'well_type': parts[0],
        #         'oil_prod': self._safe_float(parts[4]),
        #         'water_prod': self._safe_float(parts[5]),
        #         'gas_prod': self._safe_float(parts[6]),
        #         'res_vol_prod': self._safe_float(parts[7]),
        #         'oil_inj': self._safe_float(parts[8]),
        #         'water_inj': self._safe_float(parts[9]),
        #         'gas_inj': self._safe_float(parts[10]),
        #         'res_vol_inj': self._safe_float(parts[11])
        #     }
        if parts[0] in ('INJ', 'PROD'):
            return {
                'day': day,
                'well_type': parts[0],
                'oil_prod': parts[4],
                'water_prod': parts[5],
                'gas_prod': parts[6],
                'res_vol_prod': parts[7],
                'oil_inj': parts[8],
                'water_inj': parts[9],
                'gas_inj': parts[10],
                'res_vol_inj': parts[11]
            }
        return None
    
    def _safe_float(self, value: str) -> float:
        """save transform to float type"""
        try:
            # ai part ctrl+c ctrl+v
            clean_value = re.sub(r'[^\d.,\-eE]', '', str(value))
            clean_value = clean_value.replace(',', '.')
            return float(clean_value) if clean_value else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _extract_iteration_number(self, file_path: Path) -> int:
        """extract iteration information for iteration column"""
        match = re.search(r'ITER_(\d+)', file_path.stem)
        return int(match.group(1))
    
    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """df -> CSV"""
        output_path = self.base_path / filename
        df.to_csv(output_path, index=False)
        print('created csv: ', output_path)


if __name__ == "__main__":
    loader = YDataFrameLoader()
    df = loader.load_final_df()
    
    print('\nfirst 10 rows: ')
    print(df.head(10))
    loader.save_to_csv(df, filename='spe1_result.csv')