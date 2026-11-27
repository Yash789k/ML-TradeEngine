export const methodLabels = {
  original: 'Mixture of experts using market states',
  recent: 'Weights updated using recent forecast errors',
  adapter: 'Weights learned from stock and market data',
  fixed: 'Fixed learned weights',
  equal: 'Equal votes for the same forecasts',
  joint: 'Jointly trained mixture of experts',
  jointEqual: 'Same three trained models with equal weights',
  router: 'Pick one forecaster with reinforcement learning',
  classic: 'Classic mixture of experts',
  market: 'Hold the S&P 500 fund',
};
