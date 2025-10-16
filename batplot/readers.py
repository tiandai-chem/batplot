"""Readers for various data formats used by batplot."""

from __future__ import annotations

import numpy as np
from typing import Tuple


def read_csv_file(fname: str):
    for delim in [",", ";", "\t"]:
        try:
            data = np.genfromtxt(fname, delimiter=delim, comments="#")
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] >= 2:
                return data
        except Exception:
            continue
    raise ValueError(f"Invalid CSV format in {fname}, need at least 2 columns (x,y).")


def read_gr_file(fname: str):
    """Read a PDF .gr file (r, G(r))."""
    r_vals = []
    g_vals = []
    with open(fname, "r") as f:
        for line in f:
            ls = line.strip()
            if not ls or ls.startswith("#"):
                continue
            parts = ls.replace(",", " ").split()
            floats = []
            for p in parts:
                try:
                    floats.append(float(p))
                except ValueError:
                    break
            if len(floats) >= 2:
                r_vals.append(floats[0])
                g_vals.append(floats[1])
    if not r_vals:
        raise ValueError(f"No numeric data found in {fname}")
    return np.array(r_vals, dtype=float), np.array(g_vals, dtype=float)


def read_fullprof_rowwise(fname: str):
    with open(fname, "r") as f:
        lines = f.readlines()[1:]
    y_rows = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        y_rows.extend([float(val) for val in line.split()])
    y = np.array(y_rows)
    return y, len(lines)


def robust_loadtxt_skipheader(fname: str):
    """Skip comments/non-numeric lines and load at least 2-column numeric data."""
    data_lines = []
    with open(fname, "r") as f:
        for line in f:
            ls = line.strip()
            if not ls or ls.startswith("#"):
                continue
            floats = []
            for p in ls.replace(",", " ").split():
                try:
                    floats.append(float(p))
                except ValueError:
                    break
            if len(floats) >= 2:
                data_lines.append(ls)
    if not data_lines:
        raise ValueError(f"No numeric data found in {fname}")
    from io import StringIO
    return np.loadtxt(StringIO("\n".join(data_lines)))


def read_mpt_file(fname: str, mode: str = 'gc', mass_mg: float = None):
    """Read BioLogic .mpt file.
    
    Args:
        fname: Path to .mpt file
        mode: 'gc' for galvanostatic cycling (specific capacity vs voltage)
              'time' for time-series data (time vs voltage/current)
              'cpc' for capacity-per-cycle (cycle number vs charge/discharge capacity)
        mass_mg: Active material mass in milligrams (required for 'gc' and 'cpc' modes)
    
    Returns:
        For 'gc' mode: (specific_capacity_data, voltage_data, cycle_numbers, charge_mask, discharge_mask)
        For 'time' mode: (time_data, voltage_data, current_data)
        For 'cpc' mode: (cycle_numbers, charge_capacity, discharge_capacity, efficiency)
    """
    import re
    
    # Read header to find number of header lines
    header_lines = 0
    with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip()
        if not first_line.startswith('EC-Lab ASCII FILE'):
            raise ValueError(f"Not a valid EC-Lab .mpt file: {fname}")
        
        # Find header lines count
        for line in f:
            if line.startswith('Nb header lines'):
                match = re.search(r'Nb header lines\s*:\s*(\d+)', line)
                if match:
                    header_lines = int(match.group(1))
                    break
        if header_lines == 0:
            raise ValueError(f"Could not find header line count in {fname}")
    
    # Read the data
    data_lines = []
    column_names = []
    
    with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
        # Skip header lines
        for i in range(header_lines - 1):
            f.readline()
        
        # Read column names (should be at header_lines - 1)
        header_line = f.readline().strip()
        column_names = [col.strip() for col in header_line.split('\t')]
        
        # Read data lines
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Replace comma decimal separator with period (European locale support)
                values = [float(val.replace(',', '.')) for val in line.split('\t')]
                if len(values) == len(column_names):
                    data_lines.append(values)
            except ValueError:
                continue
    
    if not data_lines:
        raise ValueError(f"No valid data found in {fname}")
    
    # Convert to numpy array
    data = np.array(data_lines)
    
    # Create column index mapping
    col_map = {name: i for i, name in enumerate(column_names)}
    
    if mode == 'gc':
        # Galvanostatic cycling: use BioLogic's Q charge and Q discharge columns
        if mass_mg is None or mass_mg <= 0:
            raise ValueError("Mass loading (in mg) is required and must be positive for GC mode. Use --mass parameter.")

        mass_g = float(mass_mg) / 1000.0

        # Skip first line of data as requested
        data = data[1:]

        # Required columns - try common variations
        voltage_col = col_map.get('Ewe/V', None)
        if voltage_col is None:
            voltage_col = col_map.get('Ewe', None)
        
        q_charge_col = col_map.get('Q charge/mA.h', None)
        if q_charge_col is None:
            q_charge_col = col_map.get('Q charge/mAh', None)
        
        q_discharge_col = col_map.get('Q discharge/mA.h', None)
        if q_discharge_col is None:
            q_discharge_col = col_map.get('Q discharge/mAh', None)
        
        if voltage_col is None:
            available = ', '.join(f"'{c}'" for c in column_names)
            raise ValueError(f"Could not find 'Ewe/V' or 'Ewe' column for voltage.\nAvailable columns: {available}")
        if q_charge_col is None:
            available = ', '.join(f"'{c}'" for c in column_names)
            raise ValueError(f"Could not find 'Q charge/mA.h' or 'Q charge/mAh' column.\nAvailable columns: {available}")
        if q_discharge_col is None:
            available = ', '.join(f"'{c}'" for c in column_names)
            raise ValueError(f"Could not find 'Q discharge/mA.h' or 'Q discharge/mAh' column.\nAvailable columns: {available}")

        voltage = data[:, voltage_col]
        q_charge = data[:, q_charge_col]
        q_discharge = data[:, q_discharge_col]
        
        n = len(voltage)
        
        # Determine if experiment starts with charge or discharge
        # by checking which Q column increases first
        starts_with_charge = None
        for i in range(min(100, n - 1)):
            if q_charge[i+1] > q_charge[i] + 1e-6:
                starts_with_charge = True
                break
            elif q_discharge[i+1] > q_discharge[i] + 1e-6:
                starts_with_charge = False
                break
        
        if starts_with_charge is None:
            # Default to charge if no clear increase detected
            starts_with_charge = True
        
        # Detect charge/discharge segments based on when Q values drop to 0
        # The end of charge is when Q charge drops to ~0
        # The end of discharge is when Q discharge drops to ~0
        is_charge = np.zeros(n, dtype=bool)
        
        # Set initial state
        current_is_charge = starts_with_charge
        is_charge[0] = current_is_charge
        
        # Detect segment boundaries by finding where Q values reset to ~0
        for i in range(1, n):
            if current_is_charge:
                # We're in a charge segment
                # End of charge is when Q charge drops to near 0
                if q_charge[i] < 1e-10 and q_charge[i-1] > 1e-6:
                    # Q charge just dropped to 0, switch to discharge
                    current_is_charge = False
            else:
                # We're in a discharge segment
                # End of discharge is when Q discharge drops to near 0
                if q_discharge[i] < 1e-10 and q_discharge[i-1] > 1e-6:
                    # Q discharge just dropped to 0, switch to charge
                    current_is_charge = True
            
            is_charge[i] = current_is_charge
        
        # Find charge/discharge segment boundaries
        run_starts = [0]
        for k in range(1, n):
            if is_charge[k] != is_charge[k-1]:
                run_starts.append(k)
        run_starts.append(n)
        
        # Create masks
        charge_mask = is_charge
        discharge_mask = ~is_charge
        
        # Calculate specific capacity for each segment, starting from 0
        specific_capacity = np.zeros(n, dtype=float)
        
        for r in range(len(run_starts) - 1):
            start_idx = run_starts[r]
            end_idx = run_starts[r + 1]
            
            if is_charge[start_idx]:
                # Use Q charge column
                q_values = q_charge[start_idx:end_idx]
            else:
                # Use Q discharge column
                q_values = q_discharge[start_idx:end_idx]
            
            # Reset capacity to start from 0 for this segment
            q_start = q_values[0]
            specific_capacity[start_idx:end_idx] = (q_values - q_start) / mass_g
        
        # Assign cycle numbers: each full charge-discharge or discharge-charge pair is one cycle
        cycle_numbers = np.zeros(n, dtype=int)
        current_cycle = 1
        half_cycle = 0  # Track if we're on first or second half of cycle
        
        for r in range(len(run_starts) - 1):
            start_idx = run_starts[r]
            end_idx = run_starts[r + 1]
            
            cycle_numbers[start_idx:end_idx] = current_cycle
            
            half_cycle += 1
            if half_cycle == 2:
                # Completed one full cycle (charge+discharge or discharge+charge)
                current_cycle += 1
                half_cycle = 0

        return (specific_capacity, voltage, cycle_numbers, charge_mask, discharge_mask)
    
    elif mode == 'time':
        # Time series: time vs voltage/current
        time_col = col_map.get('time/s', None)
        voltage_col = col_map.get('Ewe/V', None)
        if voltage_col is None:
            voltage_col = col_map.get('Ewe', None)
        current_col = col_map.get('<I>/mA', None)
        
        if time_col is None:
            available = ', '.join(f"'{c}'" for c in column_names)
            raise ValueError(f"Could not find 'time/s' column.\nAvailable columns: {available}")
        if voltage_col is None:
            available = ', '.join(f"'{c}'" for c in column_names)
            raise ValueError(f"Could not find 'Ewe/V' or 'Ewe' column.\nAvailable columns: {available}")
        if current_col is None:
            available = ', '.join(f"'{c}'" for c in column_names)
            raise ValueError(f"Could not find '<I>/mA' column.\nAvailable columns: {available}")
        
        time_data = data[:, time_col]
        voltage_data = data[:, voltage_col]
        current_data = data[:, current_col]
        
        return (time_data, voltage_data, current_data)
    
    elif mode == 'cv':
        # Cyclic voltammetry: voltage vs current, split by cycle
        voltage_col = col_map.get('Ewe/V', None)
        if voltage_col is None:
            voltage_col = col_map.get('Ewe', None)
        current_col = col_map.get('<I>/mA', None)
        cycle_col = col_map.get('cycle number', None)
        
        if voltage_col is None:
            available = ', '.join(f"'{c}'" for c in column_names)
            raise ValueError(f"Could not find 'Ewe/V' or 'Ewe' column for voltage.\nAvailable columns: {available}")
        if current_col is None:
            available = ', '.join(f"'{c}'" for c in column_names)
            raise ValueError(f"Could not find '<I>/mA' column for current.\nAvailable columns: {available}")
        
        voltage = data[:, voltage_col]
        current = data[:, current_col]
        if cycle_col is not None:
            cycles = data[:, cycle_col].astype(int)
        else:
            cycles = np.ones(len(voltage), dtype=int)
        return voltage, current, cycles
    elif mode == 'cpc':
        # Capacity-per-cycle: extract end-of-segment charge/discharge capacities and efficiency
        if mass_mg is None or mass_mg <= 0:
            raise ValueError("Mass loading (mg) is required and must be positive for CPC mode. Use --mass.")

        mass_g = float(mass_mg) / 1000.0

        # Skip first line of data
        data = data[1:]

        # Required columns - try common variations
        q_charge_col = col_map.get('Q charge/mA.h', None)
        if q_charge_col is None:
            q_charge_col = col_map.get('Q charge/mAh', None)
        
        q_discharge_col = col_map.get('Q discharge/mA.h', None)
        if q_discharge_col is None:
            q_discharge_col = col_map.get('Q discharge/mAh', None)
        
        if q_charge_col is None:
            available = ', '.join(f"'{c}'" for c in column_names)
            raise ValueError(f"Could not find 'Q charge/mA.h' or 'Q charge/mAh' column.\nAvailable columns: {available}")
        if q_discharge_col is None:
            available = ', '.join(f"'{c}'" for c in column_names)
            raise ValueError(f"Could not find 'Q discharge/mA.h' or 'Q discharge/mAh' column.\nAvailable columns: {available}")

        q_charge = data[:, q_charge_col]
        q_discharge = data[:, q_discharge_col]
        
        n = len(q_charge)
        
        # Determine if experiment starts with charge or discharge
        starts_with_charge = None
        for i in range(min(100, n - 1)):
            if q_charge[i+1] > q_charge[i] + 1e-6:
                starts_with_charge = True
                break
            elif q_discharge[i+1] > q_discharge[i] + 1e-6:
                starts_with_charge = False
                break
        
        if starts_with_charge is None:
            starts_with_charge = True
        
        # Detect segment boundaries by finding where Q values reset to ~0
        is_charge = np.zeros(n, dtype=bool)
        current_is_charge = starts_with_charge
        is_charge[0] = current_is_charge
        
        for i in range(1, n):
            if current_is_charge:
                if q_charge[i] < 1e-10 and q_charge[i-1] > 1e-6:
                    current_is_charge = False
            else:
                if q_discharge[i] < 1e-10 and q_discharge[i-1] > 1e-6:
                    current_is_charge = True
            is_charge[i] = current_is_charge
        
        # Find segment boundaries
        run_starts = [0]
        for k in range(1, n):
            if is_charge[k] != is_charge[k-1]:
                run_starts.append(k)
        run_starts.append(n)
        
        # Extract end-of-segment capacities
        cyc_nums = []
        cap_charge_spec = []
        cap_discharge_spec = []
        eff_percent = []
        
        current_cycle = 1
        half_cycle = 0
        cycle_charge_cap = np.nan
        cycle_discharge_cap = np.nan
        
        for r in range(len(run_starts) - 1):
            start_idx = run_starts[r]
            end_idx = run_starts[r + 1]
            
            if is_charge[start_idx]:
                # Charge segment: get capacity at end (just before it resets)
                # Use the last valid value before segment ends
                end_cap = q_charge[end_idx - 1] if end_idx > start_idx else 0.0
                cycle_charge_cap = end_cap / mass_g
            else:
                # Discharge segment: get capacity at end
                end_cap = q_discharge[end_idx - 1] if end_idx > start_idx else 0.0
                cycle_discharge_cap = end_cap / mass_g
            
            half_cycle += 1
            if half_cycle == 2:
                # Completed one full cycle
                cyc_nums.append(current_cycle)
                cap_charge_spec.append(cycle_charge_cap)
                cap_discharge_spec.append(cycle_discharge_cap)
                
                # Calculate efficiency
                if np.isfinite(cycle_charge_cap) and cycle_charge_cap > 0 and np.isfinite(cycle_discharge_cap):
                    eff = (cycle_discharge_cap / cycle_charge_cap) * 100.0
                else:
                    eff = np.nan
                eff_percent.append(eff)
                
                # Reset for next cycle
                current_cycle += 1
                half_cycle = 0
                cycle_charge_cap = np.nan
                cycle_discharge_cap = np.nan

        return (np.array(cyc_nums, dtype=float),
                np.array(cap_charge_spec, dtype=float),
                np.array(cap_discharge_spec, dtype=float),
                np.array(eff_percent, dtype=float))

    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'gc', 'time', or 'cpc'.")


def read_biologic_txt_file(fname: str, mode: str = 'cv') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read BioLogic tab-separated text export (simplified format without EC-Lab header).
    
    These .txt files have a single header line with tab-separated column names,
    followed by tab-separated data rows. Common format from BioLogic EC-Lab exports.
    
    Args:
        fname: Path to .txt file
        mode: Currently only 'cv' is supported (cyclic voltammetry)
    
    Returns:
        For 'cv' mode: (voltage, current, cycles)
    """
    data_lines = []
    column_names = []
    
    with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
        # First line is the header
        header_line = f.readline().strip()
        column_names = [col.strip() for col in header_line.split('\t')]
        
        # Read data lines
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Replace comma decimal separator with period (European locale support)
                values = [float(val.replace(',', '.')) for val in line.split('\t')]
                if len(values) == len(column_names):
                    data_lines.append(values)
            except ValueError:
                continue
    
    if not data_lines:
        raise ValueError(f"No valid data found in {fname}")
    
    # Convert to numpy array
    data = np.array(data_lines)
    
    # Create column index mapping
    col_map = {name: i for i, name in enumerate(column_names)}
    
    if mode == 'cv':
        # Cyclic voltammetry: voltage vs current, split by cycle
        voltage_col = col_map.get('Ewe/V', None)
        if voltage_col is None:
            voltage_col = col_map.get('Ewe', None)
        current_col = col_map.get('<I>/mA', None)
        cycle_col = col_map.get('cycle number', None)
        
        if voltage_col is None:
            available = ', '.join(f"'{c}'" for c in column_names)
            raise ValueError(f"Could not find 'Ewe/V' or 'Ewe' column for voltage.\nAvailable columns: {available}")
        if current_col is None:
            available = ', '.join(f"'{c}'" for c in column_names)
            raise ValueError(f"Could not find '<I>/mA' column for current.\nAvailable columns: {available}")
        
        voltage = data[:, voltage_col]
        current = data[:, current_col]
        if cycle_col is not None:
            cycles = data[:, cycle_col].astype(int)
        else:
            cycles = np.ones(len(voltage), dtype=int)
        return voltage, current, cycles
    else:
        raise ValueError(f"Unknown mode '{mode}' for .txt file. Currently only 'cv' is supported.")


def read_ec_csv_file(fname: str, prefer_specific: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read a cycler-exported CSV (e.g., Neware-like) and extract arrays for GC plotting.

    The CSV is expected to have a two-line header where the second header line starts
    with an empty first cell. Columns of interest include:
      - 'Voltage(V)'
      - 'Current(mA)'
      - 'Cycle Index'
      - Capacity columns:
          * absolute: 'Capacity(mAh)' and optionally split 'Chg. Cap.(mAh)' / 'DChg. Cap.(mAh)'
          * specific: 'Spec. Cap.(mAh/g)', 'Chg. Spec. Cap.(mAh/g)', 'DChg. Spec. Cap.(mAh/g)'

    Args:
        fname: Path to the CSV file.
        prefer_specific: If True, will use specific capacities (mAh g⁻¹) if available; otherwise
                         uses absolute capacities (mAh) when present.

    Returns:
        (capacity_x, voltage, cycle_numbers, charge_mask, discharge_mask)
        capacity_x: per-point capacity to plot on X (mAh or mAh g⁻¹ depending on availability/flag)
        voltage:    voltage in V
        cycle_numbers: integer cycle index; if not present in CSV, inferred by pairing alternating
                       voltage-trend segments so that Cycle 1 = first segment (charge or discharge)
                       followed by the next segment, Cycle 2 = third+fourth segments, etc.
        charge_mask: boolean mask where voltage trend is increasing (charging)
        discharge_mask: boolean mask where voltage trend is decreasing (discharging)
    """
    import csv

    # Read first two rows to compose header
    with open(fname, newline='', encoding='utf-8', errors='ignore') as f:
        r = csv.reader(f)
        try:
            r1 = next(r)
            r2 = next(r)
        except StopIteration:
            raise ValueError(f"CSV '{fname}' is empty or missing header rows")
        # If second line begins with an empty first cell, treat it as a continuation of header.
        if len(r2) > 0 and (r2[0] == '' or str(r2[0]).strip() == ''):
            header = [c.strip() for c in r1] + [c.strip() for c in r2[1:]]
            rows = list(r)
        else:
            # Single-line header: r2 is the first data row; include it back in rows
            header = [c.strip() for c in r1]
            rows = [r2] + list(r)

    # Build fast name->index map (case-insensitive match on exact header text)
    name_to_idx = {h: i for i, h in enumerate(header)}

    def _find(name: str):
        return name_to_idx.get(name, None)

    # Required columns
    v_idx = _find('Voltage(V)')
    i_idx = _find('Current(mA)')
    cyc_idx = _find('Cycle Index')
    step_type_idx = _find('Step Type')  # Optional: explicitly indicates charge/discharge
    if v_idx is None or i_idx is None:
        raise ValueError("CSV missing required 'Voltage(V)' or 'Current(mA)' columns")

    # Capacity columns (absolute preferred unless prefer_specific True)
    cap_abs_idx = _find('Capacity(mAh)')
    cap_abs_chg_idx = _find('Chg. Cap.(mAh)')
    cap_abs_dch_idx = _find('DChg. Cap.(mAh)')
    cap_spec_idx = _find('Spec. Cap.(mAh/g)')
    cap_spec_chg_idx = _find('Chg. Spec. Cap.(mAh/g)')
    cap_spec_dch_idx = _find('DChg. Spec. Cap.(mAh/g)')

    use_specific = False
    # Decide which flavor to use
    if prefer_specific and (cap_spec_chg_idx is not None or cap_spec_idx is not None):
        use_specific = True
    elif not prefer_specific and (cap_abs_chg_idx is not None or cap_abs_idx is not None):
        use_specific = False
    elif cap_abs_chg_idx is None and cap_abs_idx is None and (cap_spec_idx is not None or cap_spec_chg_idx is not None):
        use_specific = True
    # else: fallback stays False (absolute) if both missing we'll error later

    # Prepare arrays
    n = len(rows)
    voltage = np.empty(n, dtype=float)
    current = np.empty(n, dtype=float)
    cycles = np.ones(n, dtype=int)
    cap_x = np.full(n, np.nan, dtype=float)

    def _to_float(val: str) -> float:
        try:
            return float(val.strip()) if isinstance(val, str) else float(val)
        except Exception:
            return np.nan

    for k, row in enumerate(rows):
        # Ensure row has enough columns
        if len(row) < len(header):
            row = row + [''] * (len(header) - len(row))
        v = _to_float(row[v_idx])
        i = _to_float(row[i_idx])
        voltage[k] = v
        current[k] = i
        if cyc_idx is not None:
            cval = _to_float(row[cyc_idx])
            try:
                cycles[k] = int(cval) if not np.isnan(cval) else 1
            except Exception:
                cycles[k] = 1
        # Don't decide chg/dchg capacity here; we will assign after deriving direction
        # Fill combined capacity columns if present (used when split columns missing)
        if use_specific and cap_spec_idx is not None:
            cap_x[k] = _to_float(row[cap_spec_idx])
        elif (not use_specific) and cap_abs_idx is not None:
            cap_x[k] = _to_float(row[cap_abs_idx])

    # --- Derive charge/discharge from Step Type column (if available) or voltage trend ---
    # First try to use explicit Step Type column (e.g., "CC Chg", "CC DChg", "Charge", "Discharge")
    is_charge = np.zeros(n, dtype=bool)
    if step_type_idx is not None:
        # Parse Step Type to determine charge/discharge
        for k, row in enumerate(rows):
            if len(row) < len(header):
                row = row + [''] * (len(header) - len(row))
            step_type = str(row[step_type_idx]).strip().lower()
            # Check for discharge first (since "discharge" contains "charge")
            # Discharge indicators: "dchg", "dischg", "discharge", "cc dchg", etc.
            is_dchg = 'dchg' in step_type or 'dischg' in step_type or step_type.startswith('dis')
            # Then check for charge indicators: "chg", "charge", "cc chg", etc.
            is_chg = (not is_dchg) and (('chg' in step_type) or ('charge' in step_type))
            # If it's clearly charge, mark as charge; if discharge, mark as discharge
            # Otherwise (Rest, CV, etc.), use previous direction or fallback
            if is_chg:
                is_charge[k] = True
            elif is_dchg:
                is_charge[k] = False
            else:
                # For Rest/CV, inherit from previous row (or default to charge for first row)
                is_charge[k] = is_charge[k-1] if k > 0 else True
    else:
        # Fallback: derive charge/discharge by voltage trend (robust to current sign conventions)
        # Compute a tolerant derivative and propagate direction over plateaus (|dv| <= eps)
        # eps based on dynamic range to avoid noise flips
        v_clean = np.array(voltage, dtype=float)
        v_min = np.nanmin(v_clean) if np.isfinite(v_clean).any() else 0.0
        v_max = np.nanmax(v_clean) if np.isfinite(v_clean).any() else 1.0
        v_span = max(1e-6, float(v_max - v_min))
        eps = max(1e-6, 1e-4 * v_span)
        dv = np.diff(v_clean)
        dv = np.nan_to_num(dv, nan=0.0, posinf=0.0, neginf=0.0)
        # Initial direction: first significant dv sets the sign; fallback to current sign if needed
        init_dir = None
        for d in dv[: min(500, dv.size)]:
            if abs(d) > eps:
                init_dir = (d > 0)
                break
        if init_dir is None:
            # Fallback: use sign of first non-zero current; else assume charge
            nz = None
            for i_val in current:
                if abs(i_val) > 1e-12 and np.isfinite(i_val):
                    nz = (i_val >= 0)
                    break
            init_dir = True if nz is None else bool(nz)
        prev_dir = init_dir
        for k in range(n):
            dir_set = None
            # Prefer backward-looking difference to keep the last sample of a run with its run
            if k > 0:
                db = dv[k-1]
                if abs(db) > eps:
                    dir_set = (db > 0)
            # Fallback: look forward to the next informative change (for first sample of a run)
            if dir_set is None:
                j = k
                while j < n-1:
                    d = dv[j]
                    if abs(d) > eps:
                        dir_set = (d > 0)
                        break
                    j += 1
            # If still None (flat series), keep previous
            if dir_set is None:
                dir_set = prev_dir
            is_charge[k] = dir_set
            prev_dir = dir_set

    # Build run-length encoding and optionally merge very short flicker runs
    # (Only apply smoothing when using voltage trend detection, not when using explicit Step Type)
    if step_type_idx is None:
        # Smoothing logic for voltage-trend-based detection
        run_starts = [0]
        for k in range(1, n):
            if is_charge[k] != is_charge[k-1]:
                run_starts.append(k)
        run_starts.append(n)
        # Merge runs shorter than 3 samples (or 0.2% of data length, whichever larger)
        min_len = max(3, int(0.002 * n))
        if len(run_starts) >= 3:
            keep_mask = is_charge.copy()
            new_is_charge = is_charge.copy()
            for r in range(len(run_starts)-1):
                a = run_starts[r]
                b = run_starts[r+1]
                if (b - a) < min_len:
                    # Prefer to merge into previous run if exists; else next
                    if r > 0:
                        new_is_charge[a:b] = new_is_charge[a-1]
                    elif r+1 < len(run_starts)-1:
                        new_is_charge[a:b] = new_is_charge[b]
            is_charge = new_is_charge

    # Compute final run starts for cycle inference
    run_starts = [0]
    for k in range(1, n):
        if is_charge[k] != is_charge[k-1]:
            run_starts.append(k)
    run_starts.append(n)

    # Build masks from voltage trend
    charge_mask = is_charge
    discharge_mask = ~is_charge

    # Assign capacity per-point when split chg/dchg columns exist, using derived direction
    if use_specific and (cap_spec_chg_idx is not None and cap_spec_dch_idx is not None):
        for k, row in enumerate(rows):
            # Ensure row length
            if len(row) < len(header):
                row = row + [''] * (len(header) - len(row))
            cap_chg = _to_float(row[cap_spec_chg_idx])
            cap_dch = _to_float(row[cap_spec_dch_idx])
            cap_x[k] = cap_chg if is_charge[k] else cap_dch
    elif (not use_specific) and (cap_abs_chg_idx is not None and cap_abs_dch_idx is not None):
        for k, row in enumerate(rows):
            if len(row) < len(header):
                row = row + [''] * (len(header) - len(row))
            cap_chg = _to_float(row[cap_abs_chg_idx])
            cap_dch = _to_float(row[cap_abs_dch_idx])
            cap_x[k] = cap_chg if is_charge[k] else cap_dch

    # If capacity column was missing entirely, raise
    if np.all(np.isnan(cap_x)):
        raise ValueError("No usable capacity columns found in CSV (looked for 'Capacity(mAh)' or 'Spec. Cap.(mAh/g)')")

    # Replace NaNs in capacity by 0 to avoid plotting gaps within valid segments
    # but keep masks to split charge/discharge and cycles (NaN voltage gets dropped later by plotting logic)
    cap_x = np.nan_to_num(cap_x, nan=0.0)

    # --- Cycle numbering ---
    # If CSV has a 'Cycle Index' column, use those values; otherwise infer cycles by pairing
    # alternating charge/discharge runs from voltage trend
    if cyc_idx is None:
        # Infer cycles: pair alternating runs so Cycle 1 = first two runs, Cycle 2 = next two, etc.
        inferred_cycles = np.ones(n, dtype=int)
        n_runs = len(run_starts) - 1
        for r in range(n_runs):
            a = run_starts[r]
            b = run_starts[r+1]
            cyc = (r // 2) + 1
            inferred_cycles[a:b] = cyc
        cycles = inferred_cycles
    # else: keep the cycles array as-is (already populated from CSV 'Cycle Index' column)

    return cap_x, voltage, cycles, charge_mask, discharge_mask


def read_ec_csv_dqdv_file(fname: str, prefer_specific: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Read dQ/dV or dQm/dV from a cycler CSV and return arrays for plotting vs Voltage.

    Returns: (voltage, dqdv, cycles, charge_mask, discharge_mask, y_label)
    Chooses specific dQm/dV when available if prefer_specific else falls back to absolute dQ/dV.
    Cycles and masks are inferred by voltage trend, same as read_ec_csv_file.
    """
    import csv

    # Read two header rows logic identical to read_ec_csv_file
    with open(fname, newline='', encoding='utf-8', errors='ignore') as f:
        r = csv.reader(f)
        try:
            r1 = next(r)
            r2 = next(r)
        except StopIteration:
            raise ValueError(f"CSV '{fname}' is empty or missing header rows")
        if len(r2) > 0 and (r2[0] == '' or str(r2[0]).strip() == ''):
            header = [c.strip() for c in r1] + [c.strip() for c in r2[1:]]
            rows = list(r)
        else:
            header = [c.strip() for c in r1]
            rows = [r2] + list(r)

    name_to_idx = {h: i for i, h in enumerate(header)}
    def _find(name: str):
        return name_to_idx.get(name, None)

    v_idx = _find('Voltage(V)')
    dq_abs_idx = _find('dQ/dV(mAh/V)')
    dq_spec_idx = _find('dQm/dV(mAh/V.g)')
    if v_idx is None:
        raise ValueError("CSV missing required 'Voltage(V)' column for dQ/dV plot")
    if dq_abs_idx is None and dq_spec_idx is None:
        raise ValueError("CSV missing dQ/dV columns: need 'dQ/dV(mAh/V)' or 'dQm/dV(mAh/V.g)'")

    use_spec = False
    if prefer_specific and dq_spec_idx is not None:
        use_spec = True
    elif dq_abs_idx is not None:
        use_spec = False
    elif dq_spec_idx is not None:
        use_spec = True

    y_label = r'dQm/dV (mAh g$^{-1}$ V$^{-1}$)' if use_spec else r'dQ/dV (mAh V$^{-1}$)'
    n = len(rows)
    voltage = np.empty(n, dtype=float)
    dqdv = np.empty(n, dtype=float)
    current = np.zeros(n, dtype=float)

    i_idx = _find('Current(mA)')
    def _to_float(val: str) -> float:
        try:
            return float(val.strip()) if isinstance(val, str) else float(val)
        except Exception:
            return np.nan

    for k, row in enumerate(rows):
        if len(row) < len(header):
            row = row + [''] * (len(header) - len(row))
        voltage[k] = _to_float(row[v_idx])
        if use_spec:
            dqdv[k] = _to_float(row[dq_spec_idx])
        else:
            dqdv[k] = _to_float(row[dq_abs_idx])
        if i_idx is not None:
            current[k] = _to_float(row[i_idx])

    # Derive charge/discharge masks and cycles by voltage trend (reuse core from read_ec_csv_file)
    v_clean = np.array(voltage, dtype=float)
    v_min = np.nanmin(v_clean) if np.isfinite(v_clean).any() else 0.0
    v_max = np.nanmax(v_clean) if np.isfinite(v_clean).any() else 1.0
    v_span = max(1e-6, float(v_max - v_min))
    eps = max(1e-6, 1e-4 * v_span)
    dv = np.diff(v_clean)
    dv = np.nan_to_num(dv, nan=0.0, posinf=0.0, neginf=0.0)

    # Initial dir from dv or fallback to current sign
    init_dir = None
    for d in dv[: min(500, dv.size)]:
        if abs(d) > eps:
            init_dir = (d > 0)
            break
    if init_dir is None:
        nz = None
        for i_val in current:
            if abs(i_val) > 1e-12 and np.isfinite(i_val):
                nz = (i_val >= 0)
                break
        init_dir = True if nz is None else bool(nz)

    npts = n
    is_charge = np.zeros(npts, dtype=bool)
    prev_dir = init_dir
    for k in range(npts):
        dir_set = None
        # Prefer forward-looking difference so the first sample of a new run adopts the new direction
        if k < npts - 1:
            df = dv[k]
            if abs(df) > eps:
                dir_set = (df > 0)
        # If no informative forward diff at k, search further ahead
        if dir_set is None:
            j = k + 1
            while j < npts - 1:
                d = dv[j]
                if abs(d) > eps:
                    dir_set = (d > 0)
                    break
                j += 1
        # If still none, fall back to immediate backward diff, then scan backward
        if dir_set is None and k > 0:
            db = dv[k-1]
            if abs(db) > eps:
                dir_set = (db > 0)
            else:
                j = k - 1
                while j >= 0:
                    d = dv[j]
                    if abs(d) > eps:
                        dir_set = (d > 0)
                        break
                    j -= 1
        if dir_set is None:
            dir_set = prev_dir
        is_charge[k] = dir_set
        prev_dir = dir_set

    # Build runs and infer cycles (pair alternate runs)
    run_starts = [0]
    for k in range(1, npts):
        if is_charge[k] != is_charge[k-1]:
            run_starts.append(k)
    run_starts.append(npts)
    inferred_cycles = np.ones(npts, dtype=int)
    for r in range(len(run_starts)-1):
        a, b = run_starts[r], run_starts[r+1]
        cyc = (r // 2) + 1
        inferred_cycles[a:b] = cyc

    charge_mask = is_charge
    discharge_mask = ~is_charge

    return voltage, dqdv, inferred_cycles, charge_mask, discharge_mask, y_label
