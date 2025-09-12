
# ------------------------- Core Strategy -------------------------
class RSIMeanReversion(bt.Strategy):
    params = dict(
        rsi_period=14,
        bb_period=20,
        bb_dev=2.0,
        rsi_oversold=30,       
        rsi_overbought=70,      
        rsi_exit=50, 
        stop_loss=0.02,       
        close_time=time(15, 15), 
        max_ammount_per_trade = 0.10,  # 10% of portfolio per trade
    )

    def __init__(self):

        param_string = (
            f"_rsi_p={self.p.rsi_period}"
            f"_bb_p={self.p.bb_period}"
            f"_bb_d={self.p.bb_dev}"
            f"rsi_os={self.p.rsi_oversold}"
            f"rsi_ob={self.p.rsi_overbought}"
            f"_rsi_exit={self.p.rsi_exit}"
            f"_sl={self.p.stop_loss}"
            f"_max_ammount_per_trade={self.p.max_ammount_per_trade}" 
        )
        
        # Store the parameter string as an attribute
        self.param_id = param_string


        # Indicators
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
        self.bb = bt.indicators.BollingerBands(
            period=self.p.bb_period, devfactor=self.p.bb_dev
        )
        
        # For trend filtering
        self.sma50 = bt.indicators.SimpleMovingAverage(period=50)
        self.sma200 = bt.indicators.SimpleMovingAverage(period=200)
        
        # Cross flags for exit logic
        self.rsi_x_up = bt.indicators.CrossUp(self.rsi, self.p.rsi_exit)
        self.rsi_x_down = bt.indicators.CrossDown(self.rsi, self.p.rsi_exit)
 
        self.add_timer(
            when=bt.Timer.SESSION_END, # SESSION_END is defined in the data feed
            offset=-timedelta(minutes=15),  # 15 minutes before session end
            repeat=timedelta(days=1),
            weekdays=[0, 1, 2, 3, 4],  # Mon-Fri only
        )

        self.portfolio_values = []
        self.dates = []
        self.orders = {}
        self.logged_dates = set()
        self.trade_count = 0  
        self.order = None

    def prenext(self):
        #self.__log("prenext()", self.data.datetime.datetime(0)) 
        pass

    def nextstart(self):
        #self.__log("nextstart()")
        pass

    def next(self):  

        current_time = self.data.datetime.time(0)
        dt = self.data.datetime.datetime(0)

        # Skip all trading after 15:15
        if current_time >= time(15, 15):
            self.__log("next()", f"Skipping trading after 15:15. Current Date-Time: {dt}")
            return

        # To address pending order
        if self.order:
            return

        # If self.position.size > 0, it means you have a long position.
        # If self.position.size < 0, it means you have a short position.
        # If self.position.size == 0, it means you are flat (no position).

        if self.position:
            if self.position.size > 0:  # Long position

                # Exit long if RSI reaches {rsi_exit} or price hits middle BB
                if self.rsi[0] >= self.p.rsi_exit or self.data.close[0] >= self.bb.mid[0]:
                    self.order = self.sell()
                    self.__log("next()", f"{dt} : SELL order is place (Target- ref: {self.order.ref}); RSI = {self.rsi[0]:.1f} : Closing Price = {self.data.close[0]} : bb.mid = {self.bb.mid[0]} : size = {self.position.size}")
                elif self.data.close[0] <= self.position.price * (1 - self.p.stop_loss): # self.position.price represents the average entry price
                    self.order = self.sell()
                    self.__log("next()", f"{dt} : SELL order is place (SL- ref: {self.order.ref}); RSI = {self.rsi[0]:.1f} : Closing Price = {self.data.close[0]} : bb.mid = {self.bb.mid[0]} : size = {self.position.size}")
                    

            elif self.position.size < 0:  # Short position

                # Exit short if RSI reaches {rsi_exit} or price hits middle BB
                if self.rsi[0] <= self.p.rsi_exit or self.data.close[0] <= self.bb.mid[0]:
                    self.order = self.buy()
                    self.__log("next()", f"{dt} : BUY order is place (Target- ref: {self.order.ref}); RSI = {self.rsi[0]:.2f} : Closing Price = {self.data.close[0]} : bb.mid = {self.bb.mid[0]:.2f} : size = {self.position.size}")
                # Stop loss check
                elif self.data.close[0] >= self.position.price * (1 + self.p.stop_loss): 
                    self.order = self.buy() 
                    self.__log("next()", f"{dt} : BUY order is place (SL- ref: {self.order.ref}); RSI = {self.rsi[0]:.2f} : Closing Price = {self.data.close[0]} : bb.mid = {self.bb.mid[0]:.2f} : size = {self.position.size}")
                    

            else:
                self.__err_log("next()", dt, f"{self.position.size} --> IllegalStateException") 
        else: 
            # Calculate position size based on portfolio value
            size = int(self.broker.getvalue() * self.p.max_ammount_per_trade / self.data.close[0]) # Note: both cash and the market value of all open positions
            
            if size <= 0:
                raise TradingError(param_id=self.param_id, time=dt, msg=f"Position Size can't be {size}; ['broker_val:{self.broker.getvalue()}, self.data.close[0] : {self.data.close[0]}, allowed_ammount: {self.broker.getvalue() * self.p.max_ammount_per_trade}']")

            # Long entry: RSI < rsi_oversold AND price touches lower Bollinger Band
            if self.rsi[0] < self.p.rsi_oversold and self.data.low[0] <= self.bb.bot[0]:
                self.order = self.buy(size=size)
                self.__log("next()", f"{dt} : BUY order is place (NEW- ref: {self.order.ref}); RSI = {self.rsi[0]:.2f} : Low Price = {self.data.low[0]} : bb.bot = {self.bb.bot[0]:.2f} : size = {size}")
            
            # Short entry: RSI > rsi_overbought AND price touches upper Bollinger Band
            elif self.rsi[0] > self.p.rsi_overbought and self.data.high[0] >= self.bb.top[0]:
                self.order = self.sell(size=size)
                self.__log("next()", f"{dt} : SELL order is place (NEW- ref: {self.order.ref}); RSI = {self.rsi[0]:.2f} : High Price = {self.data.high[0]} : bb.top = {self.bb.top[0]:.2f} : size = {size}")
            
        
    def notify_timer(self, timer, when, *args, **kwargs):
        """Flat everything at 15:15 local time."""
        # Cancel any pending orders first
        if self.order:
            self.__log("notify_timer()", f"Cancelling pending order ref: {self.order.ref}")
            self.cancel(self.order)
        
        if self.position.size != 0:
            self.__log("notify_timer()", f"SQUARING OFF at {when}, Position: {self.position.size}")
            self.close()

    def notify_order(self, order):

        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - nothing to do; 
            return
        
        if order.status in [order.Completed]:  
            if order.isbuy():
                # This could be either opening a long or covering a short
                if self.position.size > 0:  # We're now long, so this was an opening buy 
                    self.__log("notify_order()", f"BUY (ref: {order.ref}) executed at {order.executed.price}, size = {order.executed.size}")
                else:  # We're now flat, so this was covering a short 
                    self.__log("notify_order()", f"BUY (ref: {order.ref}) (cover) executed at {order.executed.price}, size = {order.executed.size}")
                    
            elif order.issell():
                # This could be either opening a short or closing a long
                if self.position.size < 0:  # We're now short, so this was an opening sell 
                    self.__log("notify_order()", f"SELL (ref: {order.ref}) executed at {order.executed.price}, size = {order.executed.size}")
                else:  # We're now flat, so this was closing a long 
                    self.__log("notify_order()", f"SELL (ref: {order.ref}) (close) executed at {order.executed.price}, size = {order.executed.size}")
                    
            self.order = None  # Reset order tracking 

        elif order.status in [order.Canceled, order.Margin, order.Rejected]: 
            self.__log("notify_order()", "\n", ("="*20), "\n", f"Order (ref: {order.ref}) : Status: {order.status}\n", order, "\n", ("="*20), "\n")
            self.order = None 
        
    def notify_trade(self, trade):
        status_name = "Closed" if trade.isclosed else "Opened" if trade.justopened else {0:"Created",1:"Open",2:"Closed"}.get(trade.status,f"Unknown ({trade.status})")
        if trade.isclosed:
            self.trade_count += 1
        self.__log("notify_trade()", f"trade.ref: {trade.ref} :: status: {status_name}")

    def stop(self):
        self.__log("All data is processed", f"Total trades executed: {self.trade_count}")  

    def __log(self, method, *logs: object):
        if LOGGING_ENABLED:
            log_message = " ".join(str(arg) for arg in logs)
            print(f"{self.param_id} : {method} ::: {log_message}")
        

    def __err_log(self, method, time, msg): 
        if LOGGING_ENABLED:
            print(f"{self.param_id} : ERROR : {method} ::: {time} ::: {msg}")
  