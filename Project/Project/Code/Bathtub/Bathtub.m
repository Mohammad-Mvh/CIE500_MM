clear variables
clc
%% Load DEM and select valid scenarios
% Load the DEM
[DEM, R] = readgeoraster('DEM.tif');

% Manually set EPSG code (ETRS89 / UTM zone 32N)
epsg_code = 25832;

% Read the water level CSV file: [ScenarioID, Time(min), WaterLevel]
data = readmatrix('WaterLevels.csv');

% Extract unique scenario IDs
scenario_ids = unique(data(:,1));

% Group by scenario and keep only those with 193 time steps
valid_scenarios = [];
for i = 1:length(scenario_ids)
    sc_id = scenario_ids(i);
    time_steps = sum(data(:,1) == sc_id);
    if time_steps == 193
        valid_scenarios(end+1) = sc_id; %#ok<SAGROW>
    end
end

fprintf('Total scenarios with 193 time steps: %d\n', length(valid_scenarios));

% Output directory
output_dir = 'Flood_Maps';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
%% Generate bathtub inundation maps 
%% Warning: Running this section will generate 30 GB of inundation maps
% Loop through each valid scenario
%for sc = 1:length(valid_scenarios)
 %   sc_id = valid_scenarios(sc);

    % Extract data for current scenario
  %  rows = data(:,1) == sc_id;
   % scenario_data = data(rows, :);  % [ScenarioID, Time, WaterLevel]

    % Loop through time steps
%    for t = 1:size(scenario_data, 1)
 %       water_level = scenario_data(t, 3);  % third column is water elevation
  %      timestamp = scenario_data(t, 2);    % second column is time in minutes

        % Compute inundation depth
   %     flood_depth = max(0, water_level - DEM);

        % Define output filename: e.g., sc1hr0.tif
        % output_filename = sprintf('%s/sc%02dhr%04d.tif', output_dir, sc, timestamp);

        % Save as GeoTIFF
        % geotiffwrite(output_filename, flood_depth, R, 'CoordRefSysCode', epsg_code);

        % Display progress
        % fprintf('Saved: %s\n', output_filename);
    %end
%end

disp('All valid flood maps generated successfully!');

%% Generate equivalent rainfall time series (5-min)
% Get cell area in m² from DEM reference
cell_area = abs(R.CellExtentInWorldX * R.CellExtentInWorldY);
num_rows = size(DEM,1);
num_cols = size(DEM,2);
total_area = num_rows * num_cols * cell_area;

% Preallocate rainfall matrix: 193 time steps → 576 rows (193×3), 1 col per scenario
rainfall_matrix = zeros(193*3, length(valid_scenarios));
time_series = (0:5:(193*3 - 1)*5)';  % 5-minute time vector

for sc = 1:length(valid_scenarios)
    sc_id = valid_scenarios(sc);

    % Extract scenario data
    rows = data(:,1) == sc_id;
    scenario_data = data(rows, :);

    % Loop over 193 time steps
    for t = 1:193
        water_level = scenario_data(t, 3);
        flood_depth = max(0, water_level - DEM - 10); % 10 is the river depth that was previously added to surge data

        % Volume in cubic meters
        total_volume = sum(flood_depth(:), 'omitnan') * cell_area;

        % Convert to rainfall depth (mm) over full domain
        depth_mm = (total_volume / total_area) * 1000;

        % Spread across 3 timesteps (5-min intervals)
        idx = (t-1)*3 + (1:3);
        rainfall_matrix(idx, sc) = (depth_mm / 3) * ((rand() + 1)/3);
    end
end

% Combine and export to CSV
output_rain = ['Time(min)', arrayfun(@(x) sprintf('sc%02d', x), 1:length(valid_scenarios), 'UniformOutput', false)];
rainfall_csv = [time_series, rainfall_matrix];
writecell(output_rain, 'RainfallTimeSeries.csv');   % write header
writematrix(rainfall_csv, 'RainfallTimeSeries.csv', 'WriteMode', 'append');

disp('Rainfall depth time series generated and saved as RainfallTimeSeries.csv');