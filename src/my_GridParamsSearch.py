import pandas as pd
from itertools import product
import time
from typing import Dict, List, Optional

class UniversalGridSearch:
    """
    Универсальный GridSearch для любых моделей и параметров.
    Поддерживает раннюю остановку и сохраняет реальное число итераций.
    """
    
    def __init__(self, 
                 evaluate_func,
                 results_file: str = 'grid_results.csv'):
        self.evaluate = evaluate_func
        self.results_file = results_file
        self.results = []
        self.current_stage = 0
    
    def _create_model(self, model_class, params):
        """Создаёт экземпляр модели с параметрами"""
        return model_class(**params)
    
    def _get_param_combinations(self, param_grid):
        """Генерирует все комбинации параметров"""
        keys = param_grid.keys()
        values = param_grid.values()
        return [dict(zip(keys, vals)) for vals in product(*values)]
    
    def _configure_early_stopping(self, model, early_stopping):
        """Настраивает раннюю остановку и сохраняет реальное число итераций"""
        if not early_stopping:
            return
        
        patience = early_stopping.get('patience', 10)
        
        if hasattr(model, 'patience'):
            model.patience = patience
        
        if hasattr(model, 'n_iterations'):
            model._original_iterations = model.n_iterations
    
    def run_grid(self,
                 model_class,
                 model_name: str,
                 param_grid: Dict[str, List],
                 scalers: Optional[List] = None,
                 X_train=None, y_train=None, X_val=None, y_val=None,
                 max_combinations: Optional[int] = None,
                 early_stopping: Optional[Dict] = None,
                 verbose: bool = True) -> pd.DataFrame:
        """
        Запускает перебор для одной модели с ранней остановкой.
        Сохраняет реальное число итераций в метрики.
        """
        
        self.current_stage += 1
        stage_results = []

        if scalers:
            data_combinations = [(name, Xtr, Xv) for name, Xtr, Xv in scalers]
        else:
            data_combinations = [('NoScaler', X_train, X_val)]
        
        param_combinations = self._get_param_combinations(param_grid)
        
        total = len(data_combinations) * len(param_combinations)
        if max_combinations:
            total = min(total, max_combinations)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"ПЕРЕБОР: {model_name}")
            print(f"{'='*80}")
            print(f"Параметры: {list(param_grid.keys())}")
            print(f"Комбинаций: {total}")
            if early_stopping:
                print(f"Early stopping: patience={early_stopping.get('patience', 10)}")
        
        current = 0
        start_time = time.time()
        early_stopped_count = 0
        
        for scaler_name, Xtr, Xv in data_combinations:
            for params in param_combinations:
                current += 1
                if max_combinations and current > max_combinations:
                    break
                
                # Прогресс
                if verbose and current % max(1, total//10) == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / current) * (total - current)
                    print(f"\r  [{current}/{total}] {elapsed/60:.1f} мин, ост. {eta/60:.1f} мин | ES: {early_stopped_count}", end="")
                
                try:
                    model = self._create_model(model_class, params)

                    self._configure_early_stopping(model, early_stopping)
                    
                    full_name = f"{model_name} | {scaler_name} | {params}"

                    metrics = self.evaluate(
                        model, Xtr, y_train, Xv, y_val, full_name
                    )
                    
                    actual_iterations = None
                    early_stopped = False
                    
                    if hasattr(model, 'loss_history'):
                        actual_iterations = len(model.loss_history)
                    elif hasattr(model, '_original_iterations'):
                        actual_iterations = model._original_iterations

                    if actual_iterations and 'n_iterations' in params:
                        if actual_iterations < params['n_iterations']:
                            early_stopped = True
                            early_stopped_count += 1
                            
                            if verbose and current % max(1, total//20) == 0:
                                saved = params['n_iterations'] - actual_iterations
                                print(f"\n  ⏱️ ES: {actual_iterations}/{params['n_iterations']} итер (экономия {saved} итер)")
                    
                    metrics.update({
                        'model_type': model_name,
                        'scaler': scaler_name,
                        'actual_iterations': actual_iterations,
                        'early_stopped': early_stopped,
                        **params
                    })
                    
                    stage_results.append(metrics)
                    self.results.append(metrics)
                    
                except Exception as e:
                    if verbose:
                        print(f"\nОшибка: {model_name} {params} - {str(e)[:100]}")
                    continue
        
        # Сохраняем
        df_stage = pd.DataFrame(stage_results)
        if not df_stage.empty:
            pd.DataFrame(self.results).to_csv(self.results_file, index=False)
        
        if verbose and not df_stage.empty:
            elapsed = time.time() - start_time
            print(f"\nЗавершено за {elapsed/60:.1f} мин")
            if early_stopped_count > 0:
                print(f"   Ранняя остановка: {early_stopped_count}/{total} моделей")
            
            print(f"\n🏆 ТОП-5 {model_name}:")
            display_cols = ['r2_val', 'mae_val', 'scaler', 'actual_iterations'] + [k for k in param_grid.keys()]
            top5 = df_stage.nlargest(5, 'r2_val')[display_cols]
            print(top5.to_string(index=False))
        
        return df_stage
    
    def run_sequential(self, 
                       experiments: List[Dict],
                       X_train, y_train, X_val, y_val,
                       scalers: Optional[List] = None) -> Dict[str, pd.DataFrame]:
        """Запускает несколько экспериментов последовательно"""
        results = {}
        
        for i, exp in enumerate(experiments, 1):
            print(f"\n{'#'*80}")
            print(f"ЭКСПЕРИМЕНТ {i}/{len(experiments)}: {exp.get('model_name', 'Unknown')}")
            print(f"{'#'*80}")
            
            df = self.run_grid(
                model_class=exp['model_class'],
                model_name=exp['model_name'],
                param_grid=exp['param_grid'],
                scalers=exp.get('scalers', scalers),
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                max_combinations=exp.get('max_combinations'),
                early_stopping=exp.get('early_stopping'),
                verbose=True
            )
            
            results[exp['model_name']] = df
        
        return results
    
    def get_best(self, model_type: Optional[str] = None, n: int = 5) -> pd.DataFrame:
        """Возвращает лучшие n моделей"""
        df = pd.DataFrame(self.results)
        if model_type:
            df = df[df['model_type'] == model_type]
        return df.nlargest(n, 'r2_val')[['model', 'r2_val', 'mae_val', 'scaler', 'fit_time', 'actual_iterations']]
    
    def get_best_by_scaler(self, scaler_name: Optional[str] = None, n: int = 3) -> pd.DataFrame:
        """Лучшие по каждому скалеру"""
        df = pd.DataFrame(self.results)
        if scaler_name:
            df = df[df['scaler'] == scaler_name]
        return df.nlargest(n, 'r2_val')[['model_type', 'r2_val', 'mae_val', 'scaler', 'actual_iterations']]
    
    def get_efficiency_report(self) -> pd.DataFrame:
        """Отчёт об эффективности ранней остановки"""
        df = pd.DataFrame(self.results)
        if 'actual_iterations' not in df.columns or 'n_iterations_planned' not in df.columns:
            return pd.DataFrame()
        
        df['saved_iterations'] = df['n_iterations_planned'] - df['actual_iterations']
        df['efficiency'] = (df['saved_iterations'] / df['n_iterations_planned'] * 100).fillna(0)
        
        report = df.groupby('model_type').agg({
            'actual_iterations': 'mean',
            'saved_iterations': 'mean',
            'efficiency': 'mean',
            'early_stopped': 'sum'
        }).round(1)
        
        return report
    
    def summary(self) -> pd.DataFrame:
        """Сводка по лучшим моделям каждого типа"""
        df = pd.DataFrame(self.results)
        summary = []
        for mt in df['model_type'].unique():
            subset = df[df['model_type'] == mt]
            if not subset.empty:
                best = subset.nlargest(1, 'r2_val').iloc[0]
                summary.append({
                    'Тип': mt,
                    'Лучший R²': best['r2_val'],
                    'MAE': best['mae_val'],
                    'Скалер': best['scaler'],
                    'Итераций': best.get('actual_iterations', 'N/A')
                })
        return pd.DataFrame(summary).sort_values('Лучший R²', ascending=False)