When buying calls for a long (directional) trade, the strike price selection determines your probability of profit (delta), time decay (theta), and cost. The right choice balances upside potential with realistic price targets.

⸻

1. Basic Strike Price Guidelines for Long Calls

a) At-The-Money (ATM) Calls
	•	Strike ≈ Current Price
	•	Pros:
	•	Delta ~ 0.50 (50% chance of expiring ITM)
	•	Moves closely with the stock price
	•	Good balance between cost and responsiveness
	•	Cons:
	•	Still exposed to time decay
	•	More expensive than OTM calls

b) Slightly In-The-Money (ITM) Calls (1–2 strikes below current price)
	•	Delta ~ 0.60–0.70
	•	Pros:
	•	Higher probability of profit
	•	Less impacted by time decay
	•	Behaves more like stock ownership
	•	Cons:
	•	Higher upfront cost (larger premium)

c) Out-Of-The-Money (OTM) Calls (strike above current price)
	•	Delta ~ 0.30–0.40
	•	Pros:
	•	Cheaper, higher potential leverage
	•	Cons:
	•	Lower probability of profit
	•	Needs a big price move to pay off
	•	Faster theta decay

⸻

2. Practical Rule of Thumb

For momentum-based trades:
	•	Pick a strike price near or slightly ITM (delta ≥ 0.50)
→ This ensures your call appreciates $0.50+ for every $1 the stock moves up, making it easier to profit even if the stock only moves modestly.
	•	Align strike with expected target price
If you expect the stock to move from $100 to $110 within 30 days:
	•	Buy a call with strike ≤ $105, so intrinsic value is likely achieved.

⸻

3. Using Delta to Choose Strike

A delta-based approach is often better than picking random strikes:
	•	Delta 0.50–0.65 → balanced risk/reward (ATM or slightly ITM)
	•	Delta 0.70–0.80 → more conservative, behaves like stock (deep ITM)
	•	Delta <0.40 → speculative (needs strong momentum to succeed)

⸻

4. Example: AAPL at $200
	•	Buy 200 call (ATM) → $6.50 premium, delta ~0.52
	•	Buy 195 call (ITM) → $9.20 premium, delta ~0.62 (higher chance of profit)
	•	Buy 205 call (OTM) → $4.00 premium, delta ~0.42 (cheaper but needs bigger move)

⸻

5. Additional Filters
	•	Check Implied Volatility (IV): Buy calls when IV is low (cheaper premium).
	•	Avoid strikes with very low open interest (illiquidity = wide bid/ask spreads).
	•	Consider expiry (30–45 days) as we discussed — gives time for momentum to play out.

⸻

6. Risk and Exit Rules (from trade management lessons)
	•	Position sizing: risk a fixed small fraction of account equity per trade (for example, 0.5% to 1.0%).
	•	Technical invalidation stop: define the chart level that invalidates the setup at entry and exit when hit.
	•	Never hold directional long options to zero by default; cut losses early when the setup fails.
	•	Time stop: if still in the trade near 21 DTE, close or roll unless the position is clearly working.
	•	Avoid narrative attachment: if price action breaks your setup, exit regardless of conviction in the thesis.
	•	Track expectancy quality: monitor average winner vs average loser and win rate, not only total P/L.

7. Strategy Selection by Market Conditions
	•	Long call: use when you want maximum upside convexity and accept higher theta/vega sensitivity.
	•	Call debit spread: use when you want to reduce theta decay and IV crush exposure.
	•	The spread decision should be based on expected move and volatility regime, not just lower premium cost.

⸻

Implementation Checklist for the Screener
	1.	Pick the nearest 30–45 day expiry.
	2.	Select ATM or slightly ITM strikes (delta >= 0.50), with configurable min/max delta bounds.
	3.	Output premium, delta, IV, and liquidity metrics (volume/open interest, bid/ask spread).
	4.	Add optional risk controls: max account risk per trade, stop level, and time-stop reminder near 21 DTE.
