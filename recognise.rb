#!/usr/bin/env ruby

#
# Script to recognise text using Tesseract v3
#
# Usage:
# recognise.rb /path/to/input/dir \
#              /path/to/output/dir \
#              /path/to/tesseract/bin
#

require 'json'
require 'open3'
require 'fileutils'
require 'matrix'
require 'rmagick'

def process_line(file, line)
  re = /^([^\s]+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s\d+$/
  matches = re.match(line)
  return nil if matches.nil?
  char = matches[1]
  # NOTE: Coordinate system of tesseract has (0,0) in BOTTOM-LEFT not TOP-LEFT!
  # So we convert (x,y) -> (x,h-y)
  x1 = matches[2].to_i
  y1 = matches[3].to_i
  x2 = matches[4].to_i
  y2 = matches[5].to_i
  h = Magick::Image.ping(file).first.rows
  y2 = (h - y1)
  y1 = (h - y2)
  return {
    char: char,
    x1: x1,
    y1: y1,
    x2: x2,
    y2: y2,
    width: x2 - x1,
    height: y2 - y1
  }
end

def proc_files(in_dir, out_dir, tesseract_dir)
  # Pass this into stdin for darknet (i.e., all files we want to test)
  char_regions = {}
  Dir["#{in_dir}/*.jpg"].each do |file|
    unique_image_id = File.basename(file, '.jpg')
    char_regions[unique_image_id] = {
      char: { regions: [], string: '' }
    }
    cmd = %W(
      ./tesseract
      "#{file}"
      stdout
      quiet
      makebox
      --psm 8
    ).join(' ')
    puts "Running tesseract on #{file}..."
    start = Time.now
    Open3.popen2e(cmd, chdir: tesseract_dir) do |stdin, stdoe|
      stdin.close
      while line = stdoe.gets
        puts line
        data = process_line(file, line)
        next if data.nil?
        char_regions[unique_image_id][:char][:regions] << data
        char_regions[unique_image_id][:char][:string] << data[:char]
      end
    end
    char_regions[unique_image_id][:char][:elappsed_seconds] = (Time.now() - start)
  end
  puts 'Tesseract has fininshed!'
  char_regions.each do |id, hash|
    next if hash[:char][:string].empty?
    out_file = "#{out_dir}/#{id}.json"
    puts "Writing JSON to '#{id}' to '#{out_file}'..."
    json_str = JSON.dump(hash)
    fp = File.new(out_file, 'w')
    fp.write(json_str)
    fp.close
  end
end

def main
  in_dir = ARGV[0]
  raise 'Input directory missing' if in_dir.nil?

  out_dir = ARGV[1]
  raise 'Output directory missing' if out_dir.nil?

  tesseract_dir = ARGV[2]
  raise 'Path to tesseract missing' if tesseract_dir.nil?

  FileUtils.mkdir_p(out_dir) unless Dir.exist?(out_dir)

  proc_files(in_dir, out_dir, tesseract_dir)
end

main
