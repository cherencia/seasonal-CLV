import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray
from pymc.distributions.dist_math import check_parameters
from pymc.util import RandomState
from pytensor.tensor import TensorVariable
from scipy.special import expit, hyp2f1

from pymc_marketing.clv.models.basic import CLVModel
from pymc_marketing.clv.utils import to_xarray

class BetaGeoModelWithSeasonality(CLVModel):
    r"""Beta-Geometric Negative Binomial Distribution (BG/NBD) model with time-varying covariate for seasonality.

    This class extends the standard BG/NBD model to include a binary time-varying covariate representing high and low seasons.

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame containing the following columns:
            * `customer_id`: Unique customer identifier
            * `frequency`: Number of repeat purchases
            * `recency`: Time between the first and the last purchase
            * `T`: Time between the first purchase and the end of the observation period
            * `high_season`: Indicator for high season (1 if high season, 0 if low season)
    model_config: dict, optional
        Dictionary of model prior parameters:
            * `a_prior`: Shape parameter for time until dropout; defaults to `pymc.HalfFlat()`
            * `b_prior`: Shape parameter for time until dropout; defaults to `pymc.HalfFlat()`
            * `alpha_prior`: Scale parameter for time between purchases; defaults to `pymc.HalfFlat()`
            * `r_prior`: Scale parameter for time between purchases; defaults to `pymc.HalfFlat()`
            * `phi_prior`: Effect parameter for high season; defaults to `pymc.Normal(mu=0, sigma=1)`
    sampler_config: dict, optional
        Dictionary of sampler parameters. Defaults to *None*.

    Examples
    --------
    .. code-block:: python

        from pymc_marketing.clv import BetaGeoModelWithSeasonality, rfm_summary

        data = [
            [1, "2024-01-01", 0],
            [1, "2024-02-06", 1],
            [2, "2024-01-01", 0],
            [3, "2024-01-02", 0],
            [3, "2024-01-05", 1],
            [4, "2024-01-16", 0],
            [4, "2024-02-05", 1],
            [5, "2024-01-17", 0],
            [5, "2024-01-18", 0],
            [5, "2024-01-19", 1],
        ]
        raw_data = pd.DataFrame(data, columns=["id", "date", "high_season"])

        rfm_df = rfm_summary(raw_data,'id','date')

        model = BetaGeoModelWithSeasonality(
            data=rfm_df,
            model_config={
                "r_prior": {"dist": "Gamma", "kwargs": {"alpha": 0.1, "beta": 1}},
                "alpha_prior": {"dist": "Gamma", "kwargs": {"alpha": 0.1, "beta": 1}},
                "a_prior": {"dist": "Gamma", "kwargs": {"alpha": 0.1, "beta": 1}},
                "b_prior": {"dist": "Gamma", "kwargs": {"alpha": 0.1, "beta": 1}},
                "phi_prior": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 1}},
            },
            sampler_config={
                "draws": 1000,
                "tune": 1000,
                "chains": 2,
                "cores": 2,
            },
        )

        model.fit()
        print(model.fit_summary())

        expected_purchases = model.expected_purchases(future_t=10)
        probability_alive = model.expected_probability_alive()
        expected_purchases_new_customer = model.expected_purchases_new_customer(t=10)
    """

    _model_type = "BG/NBD with Seasonality"

    def __init__(
        self,
        data: pd.DataFrame,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
    ):
        self._validate_cols(
            data,
            required_cols=["customer_id", "frequency", "recency", "T", "high_season"],
            must_be_unique=["customer_id"],
        )
        super().__init__(
            data=data,
            model_config=model_config,
            sampler_config=sampler_config,
        )

    @property
    def default_model_config(self) -> dict[str, dict]:
        return {
            "a_prior": {"dist": "HalfFlat", "kwargs": {}},
            "b_prior": {"dist": "HalfFlat", "kwargs": {}},
            "alpha_prior": {"dist": "HalfFlat", "kwargs": {}},
            "r_prior": {"dist": "HalfFlat", "kwargs": {}},
            "phi_prior": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 1}},
        }

    def build_model(self) -> None:  # type: ignore[override]
        a_prior = self._create_distribution(self.model_config["a_prior"])
        b_prior = self._create_distribution(self.model_config["b_prior"])
        alpha_prior = self._create_distribution(self.model_config["alpha_prior"])
        r_prior = self._create_distribution(self.model_config["r_prior"])
        phi_prior = self._create_distribution(self.model_config["phi_prior"])

        coords = {"customer_id": self.data["customer_id"]}
        with pm.Model(coords=coords) as self.model:
            a = self.model.register_rv(a_prior, name="a")
            b = self.model.register_rv(b_prior, name="b")

            alpha = self.model.register_rv(alpha_prior, name="alpha")
            r = self.model.register_rv(r_prior, name="r")
            phi = self.model.register_rv(phi_prior, name="phi")

            def logp(t_x, x, a, b, r, alpha, T, high_season, phi):
                x_non_zero = x > 0

                # Adjustment for high season
                lambda_high = pt.exp(phi)
                A = pt.switch(high_season, lambda_high, 1)

                D = pt.cumsum(A)

                d1 = (
                    pt.gammaln(r + x)
                    - pt.gammaln(r)
                    + pt.gammaln(a + b)
                    + pt.gammaln(b + x)
                    - pt.gammaln(b)
                    - pt.gammaln(a + b + x)
                )

                d2 = r * pt.log(alpha) - (r + x) * pt.log(alpha + t_x)
                c3 = ((alpha + t_x) / (alpha + T)) ** (r + x)
                c4 = a / (b + x - 1)

                logp = d1 + d2 + pt.log(c3 + pt.switch(x_non_zero, c4, 0))

                return check_parameters(
                    logp,
                    a > 0,
                    b > 0,
                    alpha > 0,
                    r > 0,
                    msg="a, b, alpha, r > 0",
                )

            pm.Potential(
                "likelihood",
                logp(
                    x=self.data["frequency"],
                    t_x=self.data["recency"],
                    a=a,
                    b=b,
                    alpha=alpha,
                    r=r,
                    T=self.data["T"],
                    high_season=self.data["high_season"],
                    phi=phi,
                ),
            )

    def _unload_params(self):
        trace = self.idata.posterior
        a = trace["a"]
        b = trace["b"]
        alpha = trace["alpha"]
        r = trace["r"]
        phi = trace["phi"]

        return a, b, alpha, r, phi

    def _extract_predictive_variables(
        self,
        data: pd.DataFrame,
        customer_varnames: Sequence[str] = (),
    ) -> xarray.Dataset:
        self._validate_cols(
            data,
            required_cols=[
                "customer_id",
                *customer_varnames,
            ],
            must_be_unique=["customer_id"],
        )

        a, b, alpha, r, phi = self._unload_params()

        customer_vars = to_xarray(
            data["customer_id"],
            *[data[customer_varname] for customer_varname in customer_varnames],
        )
        if len(customer_varnames) == 1:
            customer_vars = [customer_vars]

        return xarray.combine_by_coords(
            (
                a,
                b,
                alpha,
                r,
                phi,
                *customer_vars,
            )
        )

    def expected_purchases(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        if data is None:
            data = self.data

        if future_t is not None:
            data = data.assign(future_t=future_t)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["frequency", "recency", "T", "high_season", "future_t"]
        )
        a = dataset["a"]
        b = dataset["b"]
        alpha = dataset["alpha"]
        r = dataset["r"]
        phi = dataset["phi"]
        x = dataset["frequency"]
        t_x = dataset["recency"]
        T = dataset["T"]
        high_season = dataset["high_season"]
        t = dataset["future_t"]

        lambda_high = np.exp(phi)
        A = np.where(high_season, lambda_high, 1)
        D = np.cumsum(A)

        numerator = 1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp2f1(
            r + x,
            b + x,
            a + b + x - 1,
            t / (alpha + T + t),
        )
        numerator *= (a + b + x - 1) / (a - 1)
        denominator = 1 + (x > 0) * (a / (b + x - 1)) * (
            (alpha + T) / (alpha + t_x)
        ) ** (r + x)

        return (numerator / denominator).transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_probability_alive(
        self,
        data: pd.DataFrame | None = None,
    ) -> xarray.DataArray:
        if data is None:
            data = self.data

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["frequency", "recency", "T", "high_season"]
        )
        a = dataset["a"]
        b = dataset["b"]
        alpha = dataset["alpha"]
        r = dataset["r"]
        phi = dataset["phi"]
        x = dataset["frequency"]
        t_x = dataset["recency"]
        T = dataset["T"]
        high_season = dataset["high_season"]

        lambda_high = np.exp(phi)
        A = np.where(high_season, lambda_high, 1)
        D = np.cumsum(A)

        log_div = (r + x) * np.log((alpha + T) / (alpha + t_x)) + np.log(
            a / (b + np.maximum(x, 1) - 1)
        )

        return xarray.where(x == 0, 1.0, expit(-log_div)).transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def expected_purchases_new_customer(
        self,
        data: pd.DataFrame | None = None,
        *,
        t: np.ndarray | pd.Series,
    ) -> xarray.DataArray:
        if data is None:
            data = self.data

        if t is not None:
            data = data.assign(t=t)

        dataset = self._extract_predictive_variables(data, customer_varnames=["t"])
        a = dataset["a"]
        b = dataset["b"]
        alpha = dataset["alpha"]
        r = dataset["r"]
        phi = dataset["phi"]
        t = dataset["t"]

        lambda_high = np.exp(phi)
        first_term = (a + b - 1) / (a - 1)
        second_term = 1 - (alpha / (alpha + t)) ** r * hyp2f1(
            r, b, a + b - 1, t / (alpha + t)
        )

        return (first_term * second_term).transpose(
            "chain", "draw", "customer_id", missing_dims="ignore"
        )

    def _distribution_new_customers(
        self,
        random_seed: RandomState | None = None,
        var_names: Sequence[str] = ("population_dropout", "population_purchase_rate"),
    ) -> xarray.Dataset:
        with pm.Model():
            a = pm.HalfFlat("a")
            b = pm.HalfFlat("b")
            alpha = pm.HalfFlat("alpha")
            r = pm.HalfFlat("r")
            phi = pm.Normal("phi", mu=0, sigma=1)

            fit_result = self.fit_result
            if fit_result.sizes["chain"] == 1 and fit_result.sizes["draw"] == 1:
                fit_result = self.fit_result.squeeze("draw").expand_dims(
                    draw=range(1000)
                )

            pm.Beta("population_dropout", alpha=a, beta=b)
            pm.Gamma("population_purchase_rate", alpha=r, beta=alpha)

            return pm.sample_posterior_predictive(
                fit_result,
                var_names=var_names,
                random_seed=random_seed,
            ).posterior_predictive

    def distribution_new_customer_dropout(
        self,
        random_seed: RandomState | None = None,
    ) -> xarray.Dataset:
        return self._distribution_new_customers(
            random_seed=random_seed,
            var_names=["population_dropout"],
        )["population_dropout"]

    def distribution_new_customer_purchase_rate(
        self,
        random_seed: RandomState | None = None,
    ) -> xarray.Dataset:
        return self._distribution_new_customers(
            random_seed=random_seed,
            var_names=["population_purchase_rate"],
        )["population_purchase_rate"]






