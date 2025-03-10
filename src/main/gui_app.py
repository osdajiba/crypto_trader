# src/main/gui_app.py

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging
import queue
import tkinter.font as tkFont

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Custom logging handler to redirect logs to GUI
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)

class TradingSystemGUI:
    def __init__(self, root, config, args=None):
        self.root = root
        self.config = config
        self.args = args
        self.running = False
        self.thread = None
        self.log_queue = queue.Queue()
        
        # Set up the window
        self.root.title("Trading System")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Create a custom font
        self.default_font = tkFont.nametofont("TkDefaultFont")
        self.header_font = tkFont.Font(
            family=self.default_font.cget("family"),
            size=self.default_font.cget("size") + 2,
            weight="bold"
        )
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Setup tabs
        self.setup_tab = ttk.Frame(self.notebook)
        self.log_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.setup_tab, text="Setup")
        self.notebook.add(self.log_tab, text="Logs")
        self.notebook.add(self.results_tab, text="Results")
        
        # Create content for each tab
        self._create_setup_tab()
        self._create_log_tab()
        self._create_results_tab()
        
        # Initialize form values from args if provided
        if args:
            self._init_from_args(args)
            
        # Set up logging to GUI
        self._setup_logging()
        
        # Set up periodic tasks
        self.root.after(100, self._process_log_queue)
        
    def _setup_logging(self):
        """Set up logging to redirect to GUI"""
        # Get the root logger
        root_logger = logging.getLogger()
        
        # Create and add our queue handler
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setLevel(logging.INFO)  # Set appropriate level
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        queue_handler.setFormatter(formatter)
        root_logger.addHandler(queue_handler)
        
    def _process_log_queue(self):
        """Process log messages from the queue"""
        try:
            while True:
                record = self.log_queue.get_nowait()
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
                formatted_message = formatter.format(record)
                
                # Add color based on log level
                tag = "info"
                if record.levelno >= logging.ERROR:
                    tag = "error"
                elif record.levelno >= logging.WARNING:
                    tag = "warning"
                elif record.levelno >= logging.DEBUG:
                    tag = "debug"
                
                # Insert message with appropriate tag
                self.log_text.configure(state="normal")
                self.log_text.insert(tk.END, formatted_message + "\n", tag)
                self.log_text.see(tk.END)
                self.log_text.configure(state="disabled")
                
                # Process all available log records
                self.log_queue.task_done()
        except queue.Empty:
            # No more items in the queue
            pass
        finally:
            # Schedule to check again
            self.root.after(100, self._process_log_queue)
            
    def _init_from_args(self, args):
        """Initialize form values from command line args"""
        if args.config:
            self.config_var.set(args.config)
        
        if args.mode:
            self.mode_var.set(args.mode)
            
        if args.strategy:
            self.strategy_var.set(args.strategy)
            
        if args.symbol:
            self.symbol_var.set(args.symbol)
            
        if args.timeframe:
            self.timeframe_var.set(args.timeframe)
            
        if args.start_date:
            self.start_date_var.set(args.start_date)
            
        if args.end_date:
            self.end_date_var.set(args.end_date)
            
        if args.debug:
            self.debug_var.set(True)
            
        if args.db_url:
            self.db_url_var.set(args.db_url)
            
        if args.max_workers:
            self.max_workers_var.set(args.max_workers)
    
    def _create_setup_tab(self):
        """Create content for setup tab"""
        frame = ttk.Frame(self.setup_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        header = ttk.Label(frame, text="Trading System Configuration", font=self.header_font)
        header.grid(column=0, row=0, columnspan=3, pady=(0, 10), sticky=tk.W)
        
        # Create form variables
        self.config_var = tk.StringVar(value=str(Path(project_root) / "conf/bt_config.yaml"))
        self.mode_var = tk.StringVar(value="backtest")
        self.strategy_var = tk.StringVar(value=self.config.get("strategy", "active", default="dual_ma"))
        self.symbol_var = tk.StringVar(value=",".join(self.config.get("trading", "instruments", default=["BTC/USDT"])))
        self.timeframe_var = tk.StringVar(value="1h")
        
        # Use current date and one month ago as defaults for backtest period
        today = datetime.date.today()
        one_month_ago = today - datetime.timedelta(days=30)
        
        self.start_date_var = tk.StringVar(value=one_month_ago.strftime("%Y-%m-%d"))
        self.end_date_var = tk.StringVar(value=today.strftime("%Y-%m-%d"))
        self.debug_var = tk.BooleanVar(value=False)
        self.db_url_var = tk.StringVar(value=self.config.get("database", "url", default=""))
        self.max_workers_var = tk.IntVar(value=self.config.get("system", "performance", "max_threads", default=4))
        
        # Configuration file
        row = 1
        ttk.Label(frame, text="Configuration File:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        config_entry = ttk.Entry(frame, textvariable=self.config_var, width=50)
        config_entry.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        config_btn = ttk.Button(frame, text="Browse...", command=self._browse_config)
        config_btn.grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Mode selection
        row += 1
        ttk.Label(frame, text="Trading Mode:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        modes = ["backtest", "paper", "live"]
        mode_combo = ttk.Combobox(frame, textvariable=self.mode_var, values=modes, state="readonly")
        mode_combo.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Strategy selection
        row += 1
        ttk.Label(frame, text="Strategy:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        strategies = ["dual_ma", "neural_network", "other_strategy"]  # Add your strategies
        strategy_combo = ttk.Combobox(frame, textvariable=self.strategy_var, values=strategies)
        strategy_combo.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Trading symbol
        row += 1
        ttk.Label(frame, text="Trading Symbol:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.symbol_var).grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(frame, text="(comma-separated for multiple)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Timeframe
        row += 1
        ttk.Label(frame, text="Timeframe:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
        ttk.Combobox(frame, textvariable=self.timeframe_var, values=timeframes).grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Backtest period - Start date
        row += 1
        ttk.Label(frame, text="Start Date:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        start_date = ttk.Entry(frame, textvariable=self.start_date_var)
        start_date.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(frame, text="(YYYY-MM-DD)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Backtest period - End date
        row += 1
        ttk.Label(frame, text="End Date:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        end_date = ttk.Entry(frame, textvariable=self.end_date_var)
        end_date.grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(frame, text="(YYYY-MM-DD)").grid(column=2, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Database URL
        row += 1
        ttk.Label(frame, text="Database URL:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.db_url_var).grid(column=1, row=row, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Max workers
        row += 1
        ttk.Label(frame, text="Max Workers:").grid(column=0, row=row, sticky=tk.W, padx=5, pady=5)
        ttk.Spinbox(frame, from_=1, to=16, textvariable=self.max_workers_var, width=5).grid(column=1, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Debug mode
        row += 1
        ttk.Checkbutton(frame, text="Debug Mode", variable=self.debug_var).grid(column=1, row=row, sticky=tk.W, padx=5, pady=5)
        
        # Create a separator
        row += 1
        ttk.Separator(frame, orient=tk.HORIZONTAL).grid(column=0, row=row, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Create buttons
        row += 1
        button_frame = ttk.Frame(frame)
        button_frame.grid(column=0, row=row, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Start", command=self.start_trading).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop", command=self.stop_trading).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset", command=self.reset_form).pack(side=tk.LEFT, padx=5)
        
        # Make the grid resizable
        frame.columnconfigure(1, weight=1)
        
    def _create_log_tab(self):
        """Create content for log tab"""
        log_frame = ttk.Frame(self.log_tab, padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a scrolled text widget for logs
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state="disabled")
        
        # Configure tags for different log levels
        self.log_text.tag_configure("error", foreground="red")
        self.log_text.tag_configure("warning", foreground="orange")
        self.log_text.tag_configure("info", foreground="black")
        self.log_text.tag_configure("debug", foreground="gray")
        
        # Button frame
        button_frame = ttk.Frame(log_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(button_frame, text="Clear Logs", command=self._clear_logs).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Logs", command=self._save_logs).pack(side=tk.LEFT, padx=5)
        
    def _create_results_tab(self):
        """Create content for results tab"""
        results_frame = ttk.Frame(self.results_tab, padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a scrolled text widget for results
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=20)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.configure(state="disabled")
        
        # Button frame
        button_frame = ttk.Frame(results_frame)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(button_frame, text="Save Results", command=self._save_results).pack(side=tk.LEFT, padx=5)
    
    def _browse_config(self):
        """Open file dialog to select config file"""
        filename = filedialog.askopenfilename(
            title="Select Configuration File",
            filetypes=(
                ("YAML files", "*.yaml *.yml"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ),
            initialdir=Path(project_root) / "conf"
        )
        if filename:
            self.config_var.set(filename)
    
    def _clear_logs(self):
        """Clear the log display"""
        self.log_text.configure(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state="disabled")
    
    def _save_logs(self):
        """Save logs to a file"""
        filename = filedialog.asksaveasfilename(
            title="Save Logs",
            defaultextension=".log",
            filetypes=(("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*"))
        )
        if filename:
            with open(filename, "w") as f:
                f.write(self.log_text.get(1.0, tk.END))
            messagebox.showinfo("Save Logs", f"Logs saved to {filename}")
    
    def _save_results(self):
        """Save results to a file"""
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=(("Text files", "*.txt"), ("JSON files", "*.json"), ("All files", "*.*"))
        )
        if filename:
            with open(filename, "w") as f:
                f.write(self.results_text.get(1.0, tk.END))
            messagebox.showinfo("Save Results", f"Results saved to {filename}")
    
    def _run_trading_system(self):
        """Run the trading system in a separate thread"""
        try:
            # Get the logger
            logger = logging.getLogger("trading_system")
            logger.info("Starting trading system from GUI...")
            
            # Import required modules
            from src.common.async_executor import AsyncExecutor
            
            # Dynamically import the core module
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "core", 
                os.path.join(project_root, "src/core/core.py")
            )
            core_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(core_module)
            
            # Apply form values to config
            config_path = Path(self.config_var.get())
            if not config_path.exists():
                logger.error(f"Configuration file not found: {config_path}")
                return
            
            # Load config if changed
            if str(config_path) != str(self.config._config_path):
                logger.info(f"Loading new configuration from {config_path}")
                self.config.load(config_path)
            
            # Apply form values to config
            logger.info("Applying form values to configuration...")
            self.config.set("system", "operational_mode", self.mode_var.get())
            self.config.set("strategy", "active", self.strategy_var.get())
            # Split and strip symbols
            symbols = [s.strip() for s in self.symbol_var.get().split(',')]
            self.config.set("trading", "instruments", symbols)
            # Set timeframe if available in config structure
            self.config.set("strategy", "timeframe", self.timeframe_var.get())
            self.config.set("backtest", "period", "start", self.start_date_var.get())
            self.config.set("backtest", "period", "end", self.end_date_var.get())
            if self.db_url_var.get():
                self.config.set("database", "url", self.db_url_var.get())
            self.config.set("system", "performance", "max_threads", self.max_workers_var.get())
            
            # Create core instance
            TradingCore = core_module.TradingCore
            core = TradingCore(self.config)
            
            # Run the trading pipeline
            logger.info("Starting trading pipeline...")
            executor = AsyncExecutor()
            
            # Start the pipeline
            result = executor.run(core.run_pipeline())
            
            # Display results
            self.display_results(result)
            
            logger.info("Trading pipeline completed")
            
            # Shutdown properly
            executor.run(core.shutdown())
            executor.run(executor.close())
            
        except Exception as e:
            logger.error(f"Error running trading system: {str(e)}", exc_info=True)
            messagebox.showerror("Error", f"Error running trading system:\n{str(e)}")
        finally:
            self.running = False
    
    def display_results(self, result):
        """Display results in the results tab"""
        if not result:
            return
            
        self.results_text.configure(state="normal")
        self.results_text.delete(1.0, tk.END)
        
        # Format the results
        if isinstance(result, dict):
            formatted_result = json.dumps(result, indent=2)
        else:
            formatted_result = str(result)
            
        self.results_text.insert(tk.END, formatted_result)
        self.results_text.configure(state="disabled")
        
        # Switch to results tab
        self.notebook.select(self.results_tab)
    
    def start_trading(self):
        """Start the trading system"""
        if self.running:
            messagebox.showinfo("Already Running", "Trading system is already running.")
            return
            
        # Validate inputs
        try:
            # Check that dates are valid
            datetime.datetime.strptime(self.start_date_var.get(), "%Y-%m-%d")
            datetime.datetime.strptime(self.end_date_var.get(), "%Y-%m-%d")
            
            # Check that required fields are filled
            if not self.config_var.get() or not Path(self.config_var.get()).exists():
                messagebox.showerror("Invalid Config", "Please select a valid configuration file.")
                return
                
            if not self.symbol_var.get():
                messagebox.showerror("Missing Symbol", "Please specify at least one trading symbol.")
                return
        except ValueError as e:
            messagebox.showerror("Invalid Date Format", f"Please use YYYY-MM-DD format for dates: {str(e)}")
            return
            
        # Set running flag and start thread
        self.running = True
        self.thread = threading.Thread(target=self._run_trading_system)
        self.thread.daemon = True
        self.thread.start()
        
        # Switch to log tab
        self.notebook.select(self.log_tab)
    
    def stop_trading(self):
        """Stop the trading system"""
        if not self.running:
            messagebox.showinfo("Not Running", "Trading system is not running.")
            return
            
        # Log stop request
        logger = logging.getLogger("trading_system")
        logger.warning("User requested to stop trading system")
        
        # Set flag to stop
        self.running = False
        
        # Show message
        messagebox.showinfo("Stop Requested", 
                           "Stop request sent. The system will shut down after the current operation completes.")
    
    def save_config(self):
        """Save current configuration to file"""
        # Get config path
        config_path = self.config_var.get()
        if not config_path:
            config_path = filedialog.asksaveasfilename(
                title="Save Configuration",
                defaultextension=".yaml",
                filetypes=(("YAML files", "*.yaml"), ("All files", "*.*")),
                initialdir=Path(project_root) / "conf"
            )
            if not config_path:
                return
            self.config_var.set(config_path)
        
        try:
            # Apply form values to config
            self.config.set("system", "operational_mode", self.mode_var.get())
            self.config.set("strategy", "active", self.strategy_var.get())
            # Split and strip symbols
            symbols = [s.strip() for s in self.symbol_var.get().split(',')]
            self.config.set("trading", "instruments", symbols)
            # Set timeframe if available in config structure
            self.config.set("strategy", "timeframe", self.timeframe_var.get())
            self.config.set("backtest", "period", "start", self.start_date_var.get())
            self.config.set("backtest", "period", "end", self.end_date_var.get())
            if self.db_url_var.get():
                self.config.set("database", "url", self.db_url_var.get())
            self.config.set("system", "performance", "max_threads", self.max_workers_var.get())
            
            # Save config
            self.config.save(config_path)
            messagebox.showinfo("Configuration Saved", f"Configuration saved to {config_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving configuration:\n{str(e)}")
    
    def reset_form(self):
        """Reset form to default values"""
        if messagebox.askyesno("Reset Form", "Are you sure you want to reset all fields to default values?"):
            # Reset to defaults
            self.config_var.set(str(Path(project_root) / "conf/bt_config.yaml"))
            self.mode_var.set("backtest")
            self.strategy_var.set("dual_ma")
            self.symbol_var.set("BTC/USDT")
            self.timeframe_var.set("1h")
            
            # Use current date and one month ago for date range
            today = datetime.date.today()
            one_month_ago = today - datetime.timedelta(days=30)
            self.start_date_var.set(one_month_ago.strftime("%Y-%m-%d"))
            self.end_date_var.set(today.strftime("%Y-%m-%d"))
            
            self.debug_var.set(False)
            self.db_url_var.set("")
            self.max_workers_var.set(4)