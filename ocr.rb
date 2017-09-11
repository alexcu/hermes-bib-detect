#!/usr/bin/env ruby

#
# Script to recognise text using Tesseract v4 alpha
#
# Usage:
# ocr.rb /path/to/input/dir \
#        /path/to/output/dir \
#        /path/to/tesseract/bin
#

require 'json'
require 'open3'
require 'fileutils'
require 'matrix'
require 'rmagick'

CHAR_WHITELIST = "^A-Z0-9\-".freeze

def process_line(file, line)
  re = /^([^\s]+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s\d+$/
  matches = re.match(line)
  return nil if matches.nil?
  char = matches[1].tr(CHAR_WHITELIST, '')
  return nil if char.length.zero?
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
  string_regions = {}
  Dir["#{in_dir}/*.jpg"].each do |file|
    matches = File.basename(file).match(/([^\.]+)\.pp(\d)\.jpg/)
    next if matches[1].nil?
    unique_image_id = matches[1]
    p_id = matches[2].to_i
    if string_regions[unique_image_id].nil?
      string_regions[unique_image_id] = {
        ocr: []
      }
    end
    string_regions[unique_image_id][:ocr][p_id] = { regions: [], string: '' }
    # PSM of 8 recommended for text regions
    cmd = %W[
      ./tesseract
      "#{file}"
      stdout
      -psm 8
      -oem 4
      quiet
      makebox
    ].join(' ')
    puts "Running tesseract on #{file}..."
    start = Time.now
    Open3.popen2e(cmd, chdir: tesseract_dir) do |stdin, stdoe|
      stdin.close
      while line = stdoe.gets
        puts line
        data = process_line(file, line)
        next if data.nil?
        string_regions[unique_image_id][:ocr][p_id][:regions] << data
        string_regions[unique_image_id][:ocr][p_id][:string] << data[:char]
      end
    end
    string_regions[unique_image_id][:ocr][p_id][:elappsed_seconds] = (Time.now - start)
  end
  puts 'Tesseract has fininshed!'
  string_regions.each do |id, hash|
    # Any two processed ocr are the same? Merge as one.
    hash[:ocr].uniq! { |e| e[:string] }
    next if hash[:ocr].nil? || hash[:ocr].empty?
    # Only keep recognition for whitelist characters
    out_hash = hash[:ocr].reject do |proc_hash|
      proc_hash[:regions].empty? || proc_hash[:string].empty? #|| (proc_hash[:string] =~ CHAR_WHITELIST).nil?
    end
    next if out_hash.empty?
    out_hash = { ocr: out_hash } # Wrap in ocr
    out_file = "#{out_dir}/#{id}.json"
    puts "Writing JSON to '#{id}' to '#{out_file}'..."
    json_str = JSON.dump(out_hash)
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
