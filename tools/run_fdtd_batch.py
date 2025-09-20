#!/usr/bin/env python3
"""
WE-FDTD Batch Runner

Batch launcher to run multiple FDTD cases in parallel and collect logs.
This script enables running multiple WE-FDTD simulations concurrently,
useful for parameter sweeps, optimization, and batch processing.
"""

import os
import sys
import argparse
import json
import time
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from datetime import datetime

# Import our FDTD runner
sys.path.append(str(Path(__file__).parent))
from fdtd_runner import FDTDRunner


class FDTDBatchRunner:
    """
    Batch runner for multiple WE-FDTD simulations.
    
    This class manages:
    - Parallel execution of multiple FDTD cases
    - Log collection and aggregation
    - Progress monitoring
    - Result compilation
    """
    
    def __init__(self, max_workers=None, log_level=logging.INFO):
        """
        Initialize batch runner.
        
        Args:
            max_workers (int): Maximum number of parallel workers (default: CPU count)
            log_level: Logging level
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.cases = []
        self.results = {}
        self.start_time = None
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'fdtd_batch_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def add_case(self, case_id, config, fdtd_binary=None, use_mpi=False, mpi_ranks=1):
        """
        Add a simulation case to the batch.
        
        Args:
            case_id (str): Unique identifier for the case
            config (dict): FDTD configuration parameters
            fdtd_binary (str): Path to FDTD binary (optional)
            use_mpi (bool): Whether to use MPI
            mpi_ranks (int): Number of MPI ranks
        """
        case = {
            'case_id': case_id,
            'config': config,
            'fdtd_binary': fdtd_binary,
            'use_mpi': use_mpi,
            'mpi_ranks': mpi_ranks
        }
        self.cases.append(case)
        self.logger.info(f"Added case: {case_id}")
    
    def load_cases_from_file(self, cases_file):
        """
        Load simulation cases from a file.
        
        Args:
            cases_file (str): Path to cases file (JSON or text format)
        """
        cases_path = Path(cases_file)
        
        if not cases_path.exists():
            raise FileNotFoundError(f"Cases file not found: {cases_file}")
        
        if cases_path.suffix.lower() == '.json':
            self._load_json_cases(cases_path)
        else:
            self._load_text_cases(cases_path)
    
    def _load_json_cases(self, json_path):
        """Load cases from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'cases' in data:
            for case in data['cases']:
                self.add_case(
                    case_id=case['case_id'],
                    config=case['config'],
                    fdtd_binary=case.get('fdtd_binary'),
                    use_mpi=case.get('use_mpi', False),
                    mpi_ranks=case.get('mpi_ranks', 1)
                )
        else:
            raise ValueError("JSON file must contain 'cases' array")
    
    def _load_text_cases(self, text_path):
        """Load cases from text file (simple format)."""
        with open(text_path, 'r') as f:
            lines = f.readlines()
        
        case_id = 1
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Parse simple parameter format: freq=50000,source_x=0.1,source_y=0.5
                params = {}
                for param in line.split(','):
                    if '=' in param:
                        key, value = param.split('=', 1)
                        try:
                            params[key.strip()] = float(value.strip())
                        except ValueError:
                            params[key.strip()] = value.strip()
                
                self.add_case(
                    case_id=f"case_{case_id:04d}",
                    config=params
                )
                case_id += 1
    
    def run_batch(self, output_base_dir="outputs"):
        """
        Execute all cases in the batch.
        
        Args:
            output_base_dir (str): Base directory for all outputs
            
        Returns:
            dict: Batch execution results
        """
        if not self.cases:
            self.logger.warning("No cases to run")
            return {}
        
        self.start_time = time.time()
        output_base = Path(output_base_dir)
        output_base.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting batch execution of {len(self.cases)} cases")
        self.logger.info(f"Using {self.max_workers} parallel workers")
        
        # Execute cases in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_case = {}
            for case in self.cases:
                output_dir = output_base / case['case_id']
                future = executor.submit(self._run_single_case, case, str(output_dir))
                future_to_case[future] = case['case_id']
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_case):
                case_id = future_to_case[future]
                try:
                    result = future.result()
                    self.results[case_id] = result
                    completed += 1
                    
                    if result['success']:
                        self.logger.info(f"Case {case_id} completed successfully ({completed}/{len(self.cases)})")
                    else:
                        self.logger.error(f"Case {case_id} failed: {result.get('stderr', 'Unknown error')}")
                    
                except Exception as exc:
                    self.logger.error(f"Case {case_id} generated exception: {exc}")
                    self.results[case_id] = {
                        'success': False,
                        'error': str(exc),
                        'case_id': case_id
                    }
                    completed += 1
        
        self._generate_summary_report(output_base)
        return self.results
    
    def _run_single_case(self, case, output_dir):
        """
        Run a single FDTD case.
        
        Args:
            case (dict): Case configuration
            output_dir (str): Output directory for this case
            
        Returns:
            dict: Case execution results
        """
        try:
            # Create FDTD runner for this case
            runner = FDTDRunner(
                fdtd_binary_path=case.get('fdtd_binary'),
                work_dir=output_dir,
                use_mpi=case.get('use_mpi', False),
                mpi_ranks=case.get('mpi_ranks', 1)
            )
            
            # Write input file
            input_file = runner.write_input_dat(case['config'])
            
            # Run simulation
            result = runner.run_fdtd_simulation(input_file)
            result['case_id'] = case['case_id']
            result['input_file'] = input_file
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'case_id': case['case_id']
            }
    
    def _generate_summary_report(self, output_base):
        """Generate summary report of batch execution."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        summary = {
            'total_cases': len(self.cases),
            'successful_cases': sum(1 for r in self.results.values() if r.get('success', False)),
            'failed_cases': sum(1 for r in self.results.values() if not r.get('success', False)),
            'execution_time_seconds': duration,
            'cases': self.results
        }
        
        # Write summary to JSON
        summary_file = output_base / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log summary
        self.logger.info(f"Batch execution completed in {duration:.2f} seconds")
        self.logger.info(f"Successful: {summary['successful_cases']}/{summary['total_cases']}")
        self.logger.info(f"Summary report written to: {summary_file}")


def main():
    """Main command-line interface for batch runner."""
    parser = argparse.ArgumentParser(description="WE-FDTD Batch Runner")
    parser.add_argument("--cases-file", required=True, help="File containing simulation cases")
    parser.add_argument("--output-dir", default="outputs", help="Base output directory")
    parser.add_argument("--max-workers", type=int, help="Maximum number of parallel workers")
    parser.add_argument("--fdtd-binary", help="Path to WE-FDTD binary")
    parser.add_argument("--use-mpi", action="store_true", help="Use MPI for all cases")
    parser.add_argument("--mpi-ranks", type=int, default=1, help="Number of MPI ranks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Create batch runner
    batch_runner = FDTDBatchRunner(max_workers=args.max_workers, log_level=log_level)
    
    try:
        # Load cases
        batch_runner.load_cases_from_file(args.cases_file)
        
        # Override binary and MPI settings if provided
        if args.fdtd_binary or args.use_mpi:
            for case in batch_runner.cases:
                if args.fdtd_binary:
                    case['fdtd_binary'] = args.fdtd_binary
                if args.use_mpi:
                    case['use_mpi'] = True
                    case['mpi_ranks'] = args.mpi_ranks
        
        # Run batch
        results = batch_runner.run_batch(args.output_dir)
        
        # Print final summary
        successful = sum(1 for r in results.values() if r.get('success', False))
        total = len(results)
        print(f"\nBatch execution completed: {successful}/{total} cases successful")
        
        # Exit with error code if any cases failed
        sys.exit(0 if successful == total else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()