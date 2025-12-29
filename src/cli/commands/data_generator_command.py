"""Data generator command implementation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from .base_command import BaseCommand


@dataclass
class DataGenerationParams:
    """Parameters for data generation."""
    scenario: str | None = None
    source_dir: str | None = None
    output_dir: str | None = None
    source_file: str | None = None
    output_file: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    sample_size: int | None = None
    copy: bool = False
    list_files: bool = False


@dataclass 
class SineWaveParams:
    """Sine wave pattern parameters."""
    enabled: bool = False
    n_periods: int | None = None
    samples_per_period: int | None = None
    base_price: float | None = None
    amplitude: float | None = None
    trend_slope: float | None = None
    volatility: float | None = None


@dataclass
class UpwardDriftParams:
    """Upward drift pattern parameters."""
    enabled: bool = False
    drift_samples: int | None = None
    drift_rate: float | None = None
    drift_volatility: float | None = None
    drift_floor: float | None = None


@dataclass
class ScenarioDefaults:
    """Container for scenario-derived default values."""
    
    name: str | None = None
    path: Path | None = None
    pattern_type: str | None = None
    pattern: dict[str, Any] | None = None
    data: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.pattern is None:
            self.pattern = {}
        if self.data is None:
            self.data = {}


class DataGeneratorCommand(BaseCommand):
    """Command for generating synthetic price data."""
    
    def execute(
        self,
        params: DataGenerationParams,
        sine_wave: SineWaveParams,
        upward_drift: UpwardDriftParams,
        start_date: str | None = None,
    ) -> None:
        """Execute data generation command."""
        try:
            # Load scenario defaults
            defaults = self._load_scenario_defaults(params.scenario)
            
            # Resolve generation flags
            sine_wave.enabled, upward_drift.enabled = self._derive_generation_flags(
                sine_wave.enabled, upward_drift.enabled, defaults.pattern_type
            )
            
            # Initialize generator
            generator = self._create_generator(params, defaults)
            
            # Execute appropriate generation strategy
            if params.list_files:
                self._list_files(generator)
            elif sine_wave.enabled:
                self._generate_sine_wave(generator, sine_wave, defaults, params.output_file)
            elif upward_drift.enabled:
                self._generate_upward_drift(generator, upward_drift, defaults, params.output_file)
            elif params.copy:
                self._copy_data(generator, params)
            else:
                self._generate_synthetic_sample(generator, params, start_date)
                
        except Exception as e:
            self.handle_error(e, "Data generation")
    
    def _load_scenario_defaults(self, scenario: str | None) -> ScenarioDefaults:
        """Load scenario configuration from YAML."""
        if not scenario:
            return ScenarioDefaults()

        config_path = self._resolve_config_path(scenario)
        self.logger.info("Reading scenario defaults from %s", config_path)

        with config_path.open("r", encoding="utf-8") as handle:
            scenario_config = self.load_config(config_path)

        pattern_defaults = scenario_config.get("data_generator") or {}
        data_defaults = scenario_config.get("data") or {}
        raw_pattern = pattern_defaults.get("pattern_type")
        pattern_type = str(raw_pattern).lower() if raw_pattern else None

        return ScenarioDefaults(
            name=scenario,
            path=config_path,
            pattern_type=pattern_type,
            pattern=pattern_defaults,
            data=data_defaults,
        )
    
    def _resolve_config_path(self, scenario: str) -> Path:
        """Find the configuration file associated with a scenario string."""
        candidate_path = Path(scenario)

        if candidate_path.is_dir():
            candidate_path = candidate_path / "config.yaml"

        search_paths = [
            candidate_path,
            Path("src/configs/scenarios") / scenario,
            Path("src/configs/scenarios") / f"{scenario}.yaml",
        ]

        for path in search_paths:
            if path.exists():
                return path.resolve()

        raise typer.BadParameter(
            f"Scenario '{scenario}' not found. Provide a valid path or name in src/configs/scenarios."
        )
    
    def _derive_generation_flags(
        self, explicit_sine: bool, explicit_drift: bool, pattern_type: str | None
    ) -> tuple[bool, bool]:
        """Resolve which synthetic pattern to generate, combining CLI flags with scenario defaults."""
        sine_wave = explicit_sine
        upward_drift = explicit_drift

        if pattern_type == "sine_wave":
            if upward_drift and not sine_wave:
                raise typer.BadParameter(
                    "Scenario requires sine wave generation but --upward-drift was provided."
                )
            sine_wave = True
            upward_drift = False
        elif pattern_type == "upward_drift":
            if sine_wave and not upward_drift:
                raise typer.BadParameter(
                    "Scenario requires upward drift generation but --sine-wave was provided."
                )
            upward_drift = True
            sine_wave = False

        return sine_wave, upward_drift
    
    def _create_generator(self, params: DataGenerationParams, defaults: ScenarioDefaults):
        """Create and configure data generator."""
        from data_generator import PriceDataGenerator
        
        source_dir = params.source_dir or "data/raw/binance"
        output_dir, _ = self._resolve_output_targets(
            params.output_dir, params.output_file, defaults.data
        )
        output_dir = output_dir or "data/raw/synthetic"
        
        return PriceDataGenerator(source_dir=source_dir, output_dir=output_dir)
    
    def _resolve_output_targets(
        self,
        output_dir: str | None,
        output_file: str | None,
        data_defaults: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        """Derive output directory and filename from CLI args and scenario defaults."""
        if output_file is None and data_defaults.get("data_path"):
            data_path = Path(str(data_defaults["data_path"]))
            if data_path.name:
                output_file = data_path.name
            parent_dir = data_path.parent
            if output_dir is None and str(parent_dir) not in {"", "."}:
                output_dir = str(parent_dir)

        return output_dir, output_file
    
    def _coalesce(self, *values: Any) -> Any:
        """Return the first value that is not None."""
        for value in values:
            if value is not None:
                return value
        return None
    
    def _list_files(self, generator) -> None:
        """List available source files."""
        self.logger.info("Available source files:")
        source_files = generator.list_source_files()
        if not source_files:
            self.logger.warning("  No parquet files found in source directory")
        else:
            for f in source_files:
                self.logger.info("  - %s", f)
    
    def _generate_sine_wave(
        self, generator, sine_wave: SineWaveParams, defaults: ScenarioDefaults, output_file: str | None
    ) -> None:
        """Generate sine wave pattern."""
        output_file_name = output_file or "sine_wave_pattern.parquet"
        sine_params = self._collect_sine_wave_params(sine_wave, defaults.pattern)

        df = generator.generate_sine_wave_pattern(
            output_file=output_file_name,
            n_periods=int(sine_params["n_periods"]),
            samples_per_period=int(sine_params["samples_per_period"]),
            base_price=float(sine_params["base_price"]),
            amplitude=float(sine_params["amplitude"]),
            trend_slope=float(sine_params["trend_slope"]),
            volatility=float(sine_params["volatility"]),
            start_date=str(sine_params["start_date"]),
        )
        self.logger.info("Successfully generated sine wave pattern with %s rows", len(df))
    
    def _collect_sine_wave_params(
        self, sine_wave: SineWaveParams, pattern_defaults: dict[str, Any]
    ) -> dict[str, Any]:
        """Build parameter dictionary for sine wave generation."""
        return {
            "n_periods": self._coalesce(sine_wave.n_periods, pattern_defaults.get("n_periods"), 3),
            "samples_per_period": self._coalesce(
                sine_wave.samples_per_period, pattern_defaults.get("samples_per_period"), 120
            ),
            "base_price": self._coalesce(
                sine_wave.base_price, pattern_defaults.get("base_price"), 50000.0
            ),
            "amplitude": self._coalesce(sine_wave.amplitude, pattern_defaults.get("amplitude"), 5000.0),
            "trend_slope": self._coalesce(sine_wave.trend_slope, pattern_defaults.get("trend_slope"), 0.0),
            "volatility": self._coalesce(
                sine_wave.volatility, pattern_defaults.get("volatility"), 0.00001
            ),
            "start_date": self._coalesce(
                pattern_defaults.get("start_date"), "2024-01-01"
            ),
        }
    
    def _generate_upward_drift(
        self, generator, upward_drift: UpwardDriftParams, defaults: ScenarioDefaults, output_file: str | None
    ) -> None:
        """Generate upward drift pattern."""
        output_file_name = output_file or "upward_drift_pattern.parquet"
        drift_params = self._collect_upward_drift_params(upward_drift, defaults.pattern)

        df = generator.generate_upward_drift_pattern(
            output_file=output_file_name,
            n_samples=int(drift_params["n_samples"]),
            base_price=float(drift_params["base_price"]),
            drift_rate=float(drift_params["drift_rate"]),
            volatility=float(drift_params["volatility"]),
            pullback_floor=float(drift_params["pullback_floor"]),
            start_date=str(drift_params["start_date"]),
        )
        self.logger.info("Successfully generated upward drift pattern with %s rows", len(df))
    
    def _collect_upward_drift_params(
        self, upward_drift: UpwardDriftParams, pattern_defaults: dict[str, Any]
    ) -> dict[str, Any]:
        """Build parameter dictionary for upward drift generation."""
        return {
            "n_samples": self._coalesce(upward_drift.drift_samples, pattern_defaults.get("n_samples"), 500),
            "base_price": self._coalesce(
                pattern_defaults.get("base_price"), 50000.0
            ),
            "drift_rate": self._coalesce(upward_drift.drift_rate, pattern_defaults.get("drift_rate"), 0.015),
            "volatility": self._coalesce(
                upward_drift.drift_volatility, pattern_defaults.get("volatility"), 0.0005
            ),
            "pullback_floor": self._coalesce(
                upward_drift.drift_floor, pattern_defaults.get("pullback_floor"), 0.995
            ),
            "start_date": self._coalesce(
                pattern_defaults.get("start_date"), "2024-01-01"
            ),
        }
    
    def _copy_data(self, generator, params: DataGenerationParams) -> None:
        """Copy data without modifications."""
        if not params.source_file:
            raise typer.BadParameter("--source-file is required for copy operation")
        generator.copy_data(params.source_file, params.output_file)
    
    def _generate_synthetic_sample(
        self, generator, params: DataGenerationParams, start_date: str | None
    ) -> None:
        """Generate synthetic data sample."""
        if not params.source_file:
            self.logger.error(
                "Error: --source-file is required (or use --list to see available files, or provide a scenario/pattern flag)"
            )
            raise typer.Exit(1)

        df = generator.generate_synthetic_sample(
            source_file=params.source_file,
            output_file=params.output_file,
            start_date=start_date or params.start_date,
            end_date=params.end_date,
            sample_size=params.sample_size,
        )
        self.logger.info("Successfully generated synthetic data with %s rows", len(df))