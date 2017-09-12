#!/usr/bin/env ruby

require 'json'

Region = Struct.new(:x1, :y1, :x2, :y2, :rbn)

def area(region)
  w = region.x2 - region.x1
  h = region.y2 - region.y1
  w * h
end

def intersection(r1, r2)
  region = Region.new
  region.x1 = [r1.x1, r2.x1].max
  region.y1 = [r1.y1, r2.y1].max
  region.x2 = [r1.x2, r2.x2].min
  region.y2 = [r1.y2, r2.y2].min
  region if region.y1 < region.y2 && region.x1 < region.x2
end

def intersect?(r1, r2)
  !intersection(r1, r2).nil?
end

def union(r1, r2)
  region = Region.new
  region.x1 = [r1.x1, r2.x1].min
  region.y1 = [r1.y1, r2.y1].min
  region.x2 = [r1.x2, r2.x2].max
  region.y2 = [r1.y1, r2.y2].max
  region
end

def parse_estimated_bibs(json_file)
  JSON.parse(File.read(json_file))['bib']['regions'].map do |r|
    r['rbns'].map do |rbn|
      region = Region.new
      region.x1 = r['x1']
      region.y1 = r['y1']
      region.x2 = r['x2']
      region.y2 = r['y2']
      region.rbn = rbn
      region
    end
  end.flatten
end

def match_area(r1, r2)
  return 0 unless intersect?(r1, r2)
  # There are inconsistencies between match area definition.
  # I'm going to use the union one as it is referred to more.
  #(2 * area(intersection(r1, r2))) / (area(r1) + area(r2))
  area(intersection(r1, r2)).to_f / area(union(r1, r2))
end

def best_match(r, set_of_rects)
  set_of_rects.map { |r_prime| match_area(r, r_prime) }.max
end

def precision(ground_truths, estimated_bibs)
  return 0 if estimated_bibs.length.zero?
  best_matches = estimated_bibs.map { |r_e| best_match(r_e, ground_truths) }
  sum_of_estimates = best_matches.reduce(:+)
  sum_of_estimates / estimated_bibs.count
end

def recall(ground_truths, estimated_bibs)
  return 0 if estimated_bibs.length.zero?
  best_matches = ground_truths.map { |r_t| best_match(r_t, estimated_bibs) }
  sum_of_estimates = best_matches.reduce(:+)
  sum_of_estimates / ground_truths.count
end

def f_score(precision, recall, alpha = 0.5)
  1 / ((alpha / precision) + ((1 - alpha) / recall))
end

def parse_ground_truth_bibs(json_file)
  JSON.parse(File.read(json_file))['TaggedRunners'].map do |runner|
    region = Region.new
    j = 0
    rdict = runner['Bib']['PixelPoints'].each_with_index.map do |coords_str, i|
      # Only want 0,2 (two extremes)
      next unless (i % 2).zero?
      j += 1
      x, y = coords_str.split(', ')
      [["x#{j}", x.to_i], ["y#{j}", y.to_i]]
    end
    rdict = Hash[*rdict.compact.flatten]
    region.x1 = rdict['x1']
    region.y1 = rdict['y1']
    region.x2 = rdict['x2']
    region.y2 = rdict['y2']
    region.rbn = runner['Bib']['BibNumber']
    region
  end
end

def stats(ground_truths, estimated_bibs)
  # Add total number of GT bibs
  total_bibs = ground_truths.length
  # Pool of bibs in this image
  gt_rbns = ground_truths.map(&:rbn)
  estimated_rbns = estimated_bibs.map(&:rbn)
  false_positives = estimated_rbns - gt_rbns
  false_negatives = gt_rbns - estimated_rbns
  true_positives = gt_rbns & estimated_rbns
  {
    true_positives: true_positives,
    false_negatives: false_negatives,
    false_positives: false_positives,
    true_positives_rate: true_positives.length.to_f / total_bibs,
    false_negatives_rate: false_negatives.length.to_f / total_bibs,
    false_positives_rate: false_positives.length.to_f / total_bibs,
  }
end

def calc_stats(ground_truth_json, estimated_bib_json)
  ground_truths = parse_ground_truth_bibs(ground_truth_json)
  estimated_bibs = parse_estimated_bibs(estimated_bib_json)

  p = precision(ground_truths, estimated_bibs)
  r = recall(ground_truths, estimated_bibs)
  f = f_score(p, r)
  s = stats(ground_truths, estimated_bibs)
  stats_csv = "\"#{s[:true_positives].join(',')}\",\"#{s[:false_negatives].join(',')}\",\"#{s[:false_positives].join(',')}\",#{s[:true_positives_rate]},#{s[:false_negatives_rate]},#{s[:false_positives_rate]}"
  puts "#{p},#{r},#{f},#{stats_csv}"
end

def main
  ground_truth_json_dir = ARGV[0]
  estimated_bib_json_dir = ARGV[1]

  raise 'No ground truth JSON file provided' if ground_truth_json_dir.nil?
  raise 'No bib detect JSON file provided' if estimated_bib_json_dir.nil?

  puts "precision,recall,fscore,true_positives,false_negatives,false_positives,true_positives_rate,false_negatives_rate,false_positives_rate"
  Dir["#{ground_truth_json_dir}/*.json"].each do |gt_file|
    # Find respective output
    file_id = File.basename(gt_file, ".jpg.json")
    eb_file = "#{estimated_bib_json_dir}/#{file_id}.json"
    calc_stats(gt_file, eb_file) if File.exist?(eb_file)
  end
end

main
