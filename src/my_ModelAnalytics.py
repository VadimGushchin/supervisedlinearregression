import numpy as np
import pandas as pd


class ModelAnalyzer:
    """Класс для поиска лучших моделей и анализа их параметров."""

    def __init__(self, results_df, predictions_list):
        """
        Parameters:
        - results_df: DataFrame с метриками (с колонкой 'model' - объекты)
        - predictions_list: список с предсказаниями и объектами моделей
        """
        self.df = results_df.copy()
        self.df["model_str"] = self.df["model"].astype(str)
        self.predictions_by_index = predictions_list

    def _get_model_type(self, model_str):
        """
        Определяет тип модели.

        """
        if "Lasso" in model_str:
            return "Lasso"

        elif "Ridge" in model_str:
            return "Ridge"

        elif "ElasticNet" in model_str:
            return "ElasticNet"

        elif "LinearRegression" in model_str:
            return "LinearRegression"

        return "Unknown model"

    def _has_scaler(self, model_str):
        """
        Проверяет, есть ли в модели StandardScaler.

        """
        return "standardscaler" in model_str.lower()

    def find_best_by_type(self, model_type="lasso", with_scaler=None):
        """
        Находит лучшую модель по типу и наличию Scaler.

        """
        mask = self.df["model_str"].str.contains(model_type, case=False, na=False)

        if with_scaler is True:
            mask &= self.df["model_str"].str.contains(
                "standardscaler", case=False, na=False
            )

        elif with_scaler is False:
            mask &= ~self.df["model_str"].str.contains(
                "standardscaler", case=False, na=False
            )

        filtered = self.df[mask]

        if filtered.empty:
            return None

        best_idx = filtered["r2_val"].idxmax()
        best_row = filtered.loc[best_idx]

        result = {
            "model_name": str(best_row["model"]),
            "model_obj": best_row["model"],
            "r2_train": best_row["r2_train"],
            "r2_val": best_row["r2_val"],
            "mae_val": best_row["mae_val"],
            "rmse_val": best_row["rmse_val"],
            "has_scaler": self._has_scaler(str(best_row["model"])),
        }

        if model_type == "lasso":
            result["n_features"] = self._count_nonzero_features(best_row["model"])

        return result

    def _count_nonzero_features(self, model_obj):
        """
        Считает количество ненулевых весов для Lasso модели.

        """
        if hasattr(model_obj, "named_steps"):

            if "lasso" in model_obj.named_steps:
                coef = model_obj.named_steps["lasso"].coef_

            else:
                return None
        else:
            coef = getattr(model_obj, "coef_", [])

        return int(np.sum(np.abs(coef) > 1e-6))

    def get_best_ridge_df(self, with_scaler=None):
        """
        Возвращает DataFrame с лучшей Ridge моделью.

        """
        best = self.find_best_by_type("ridge", with_scaler)

        if best:
            scaler_text = "со Scaler" if best["has_scaler"] else "без Scaler"
            return pd.DataFrame(
                [
                    {
                        "Модель": f"Ridge {scaler_text}",
                        "R² train": f"{best['r2_train']:.4f}",
                        "R² val": f"{best['r2_val']:.4f}",
                        "MAE val": f"{best['mae_val']:.2f}",
                        "RMSE val": f"{best['rmse_val']:.2f}",
                    }
                ]
            )
        return pd.DataFrame()

    def get_best_lasso_df(self, with_scaler=None):
        """
        Возвращает DataFrame с лучшей Lasso моделью.

        """
        best = self.find_best_by_type("lasso", with_scaler)

        if best:
            scaler_text = "со Scaler" if best["has_scaler"] else "без Scaler"
            df = pd.DataFrame(
                [
                    {
                        "Модель": f"Lasso {scaler_text}",
                        "R² train": f"{best['r2_train']:.4f}",
                        "R² val": f"{best['r2_val']:.4f}",
                        "MAE val": f"{best['mae_val']:.2f}",
                        "RMSE val": f"{best['rmse_val']:.2f}",
                    }
                ]
            )
            if best.get("n_features"):
                df["Ненулевых весов"] = best["n_features"]
            return df
        return pd.DataFrame()

    def get_comparison_df(self, model_type="ridge"):
        """
        Возвращает DataFrame сравнения моделей с Scaler и без.

        """

        with_scaler = self.find_best_by_type(model_type, with_scaler=True)
        without_scaler = self.find_best_by_type(model_type, with_scaler=False)

        data = []
        if without_scaler:
            data.append(
                {
                    "Тип": f"{model_type.capitalize()} без Scaler",
                    "R² val": f"{without_scaler['r2_val']:.4f}",
                    "MAE val": f"{without_scaler['mae_val']:.2f}",
                }
            )

        if with_scaler:
            data.append(
                {
                    "Тип": f"{model_type.capitalize()} со Scaler",
                    "R² val": f"{with_scaler['r2_val']:.4f}",
                    "MAE val": f"{with_scaler['mae_val']:.2f}",
                }
            )

        return pd.DataFrame(data) if data else pd.DataFrame()

    def print_summary(self):
        """
        Печатает сводную таблицу всех лучших моделей.

        """
        print("\n" + "=" * 80)
        print("СВОДНАЯ ТАБЛИЦА ЛУЧШИХ МОДЕЛЕЙ")
        print("=" * 80)

        dfs = []

        dfs.append(self.get_best_ridge_df(with_scaler=True))
        dfs.append(self.get_best_ridge_df(with_scaler=False))

        dfs.append(self.get_best_lasso_df(with_scaler=True))
        dfs.append(self.get_best_lasso_df(with_scaler=False))

        summary_df = pd.concat(dfs, ignore_index=True)
        display(summary_df)

        print("\n" + "=" * 80)
        print("СРАВНЕНИЕ ВЛИЯНИЯ НОРМАЛИЗАЦИИ")
        print("=" * 80)

        ridge_comp = self.get_comparison_df("ridge")
        if not ridge_comp.empty:
            print("\nRidge регрессия:")
            display(ridge_comp)

        lasso_comp = self.get_comparison_df("lasso")
        if not lasso_comp.empty:
            print("\nLasso регрессия:")
            display(lasso_comp)
