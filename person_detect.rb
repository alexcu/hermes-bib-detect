#!/usr/bin/env ruby

#
# Script to detect people using YOLO mini on Darknet
#
# Usage:
# person_detect.rb /path/to/input/dir \
#                  /path/to/output/dir \
#                  /path/to/darknet/dir [-c]
#
# The -c parameter will crop output images of every person identified
#

require 'json'
require 'open3'
require 'fileutils'
require 'rmagick'
require 'matrix'

# The bib padding outwards value (x% of the euclidian distance)
PERSON_PADDING = 0.2
# Reject any accuracies less than this threshold
REJECT_ACCURACIES = 0.5

def euclidean_distance(p1, p2)
  sum_of_squares = 0
  p1.each_with_index do |p1_coord, index|
    sum_of_squares += (p1_coord - p2[index])**2
  end
  Math.sqrt(sum_of_squares)
end

def crop_person(in_file, out_dir, image_id, region, person_id)
  out_file = "#{out_dir}/#{image_id}_crop_person_#{person_id}.jpg"
  puts "Cropping person #{person_id} in '#{image_id}' to '#{out_file}'..."
  image = Magick::Image.read(in_file).first
  x = region[:x1]
  y = region[:y1]
  width = region[:x2] - x
  height = region[:y2] - y
  cropped_img = image.crop(x, y, width, height)
  cropped_img.write(out_file)
end

def process_line(in_dir, line, files_read)
  unique_image_id = files_read.last
  if line.include?('Predicted in')
    re = /Enter Image Path: ([^:]+): Predicted in (\d+.\d+) seconds./
    matches = re.match(line)
    filename = matches[1]
    unique_image_id = File.basename(filename, '.jpg')
    files_read << unique_image_id
    elapsed_time = matches[2].to_f
    # Yet to populate...
    return [unique_image_id, elapsed_time]
  elsif line.start_with?('person,')
    raw_data = line.split(',')
    image = Magick::Image.ping("#{in_dir}/#{unique_image_id}.jpg").first
    x1 = raw_data[2].to_i < 0 ? 0 : raw_data[2].to_i
    y1 = raw_data[3].to_i < 0 ? 0 : raw_data[3].to_i
    x2 = raw_data[4].to_i > image.columns ? image.columns : raw_data[4].to_i
    y2 = raw_data[5].to_i > image.rows ? image.rows : raw_data[5].to_i

    # Expand out its detections for padding
    scale = Matrix[[-1, -1], [+1, +1]]

    # Padding is calculated on euclidian distance of opposite diags
    diag = euclidean_distance([x1, y1], [x2, y2])
    padding = diag * PERSON_PADDING
    padded_coords = (padding * scale) + Matrix.rows([[x1, y1], [x2, y2]])

    accuracy = raw_data[1].to_f

    return if accuracy < REJECT_ACCURACIES

    return [unique_image_id, {
      accuracy: accuracy,
      x1: padded_coords[0, 0].to_i,
      y1: padded_coords[0, 1].to_i,
      x2: padded_coords[1, 0].to_i,
      y2: padded_coords[1, 1].to_i
    }]
  end
end

def proc_files(in_dir, out_dir, darknet_dir, should_crop)
  # Pass this into stdin for darknet (i.e., all files we want to test)
  files = Dir["#{in_dir}/*.jpg"]
  cmd = %w(
    ./darknet
    detector
    test
    cfg/voc.data
    cfg/tiny-yolo-voc.cfg
    tiny-yolo-voc.weights
  ).join(' ')
  puts "Running darknet on #{files.length} files... this may take a while..."
  images = {}
  Open3.popen2e(cmd, chdir: darknet_dir) do |stdin, stdoe|
    stdin.write(files.join("\n"))
    stdin.close
    while line = stdoe.gets
      puts line
      unique_image_id, data = process_line(in_dir, line, images.keys)
      # Got elapsed time
      if data.is_a?(Float)
        images[unique_image_id] = {
          person: {
            regions: [],
            elapsed_time: data
          }
        }
      # Got region data
      elsif data.is_a?(Hash)
        images[unique_image_id][:person][:regions] << data
      end
    end
  end
  puts 'Darknet has fininshed!'
  images.each do |id, hash|
    out_file = "#{out_dir}/#{id}.json"
    puts "Writing JSON to '#{id}' to '#{out_file}'..."
    json_str = JSON.dump(hash)
    fp = File.new(out_file, 'w')
    fp.write(json_str)
    fp.close
    next unless should_crop
    puts 'Cropping images to bounds...'
    hash[:person][:regions].each_with_index do |region, i|
      in_file = "#{in_dir}/#{id}.jpg"
      crop_person(in_file, out_dir, id, region, i)
    end
  end
end

def main
  in_dir = ARGV[0]
  raise 'Input directory missing' if in_dir.nil?

  out_dir = ARGV[1]
  raise 'Output directory missing' if out_dir.nil?

  darknet_dir = ARGV[2]
  raise 'Path to darknet missing' if darknet_dir.nil?

  should_crop = ARGV[3] == '-c'

  FileUtils.mkdir_p(out_dir) unless Dir.exist?(out_dir)

  proc_files(in_dir, out_dir, darknet_dir, should_crop)
end

main
