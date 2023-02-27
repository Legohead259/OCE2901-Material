function [wave_data, loc] = import_ndbc_data(station_id, start_year, end_year, path)
%   importNOAABuoyData is a function designed to ping the NOAA waves and
%   currents buoy database given a station_id and download all the wave
%   height and period data for the time specified from [start_year,
%   end_year]

% @param station_id:    The NOAA NDBC station ID
% @param start_year:    The start year of the dataset
% @param end_year:      The end year of the dataset
%
% @return wave_data: A table of all of the wave data for the years
% [start_year:end_year] including wave height and wave period
% @return loc: A location string of where the buoy is located

arguments
    station_id 
    start_year 
    end_year
    path = ""
end

switch station_id
    case "41009" % Port Canaveral Offshore
        loc = "20 NM NE of Port Canaveral, FL";
    case "41113" % Port Canaveral Nearshore
        loc = "Cape Canaveral, FL";
end

fp = sprintf("%s/%s", station_id, station_id); % EX: PATH/41009/41009

for i=start_year:end_year % For every historical year on the station
    fn = sprintf("%sh%d.txt", fp, i) % EX: "PATH/41009/41009h2003.txt"
    if isfile(fn) % Check if the data file already exists
        continue
    end

    try
        url = sprintf("https://www.ndbc.noaa.gov/view_text_file.php?filename=%sh%d.txt.gz&dir=data/historical/stdmet/", station_id, i);
        request = webread(url);
    catch
        fprintf("Failed to download buoy data for year %d. Most likely, the network is not connected, or the file does not exist.", i);
        continue
    end

    fid = fopen(fn, "w");           % Open the file for writing
    fprintf(fid, "%s", request);    % Save the NOAA buoy data to that file
    fclose(fid);                    % Close the file
end

fn = sprintf("%s_data_%d_%d.csv", fp, start_year, end_year);
if ~isfile(fn) % If the data table is not saved, then process the data
    wave_data = table();
    
    switch station_id
        case "41009" % Port Canaveral Offshore
            for i=start_year:1998 % for every year between start_year and 1998
                wave_data = vertcat(wave_data, parse_old_old_ndbc_format(sprintf("%sh%02d.txt", fp, i))); % Combine the wave data tables
            end
            for i=1999:2004 % For every year between 1999 and 2004 - In 2005, the file format changed
                wave_data = vertcat(wave_data, parse_old_ndbc_format(sprintf("%sh%02d.txt", fp, i))); % Combine the wave data tables
            end
            for i=2005:end_year % For every year between 2005 and end year
                wave_data = vertcat(wave_data, parse_ndbc_format(sprintf("%sh%02d.txt", fp, i))); % Combine the wave data tables
            end
        case "41113"
            for i=start_year:end_year % For all data years
                wave_data = vertcat(wave_data, parse_ndbc_format(sprintf("%sh%02d.txt", fp, i))); % Combine the wave data tables
            end
    end

    wave_data(ismember(wave_data.WVHT, 99.0),:) = [];
    wave_data(ismember(wave_data.WTMP, 999.0),:) = [];
    wave_data(ismember(wave_data.DPD, 99.0),:) = [];
    writetable(wave_data, fn)
else
    wave_data = readtable(fn);
end

