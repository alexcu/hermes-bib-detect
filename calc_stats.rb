#!/usr/bin/env ruby

require 'fileutils'
require 'json'
require 'csv'

Region = Struct.new(:x1, :y1, :x2, :y2, :rbn, :accuracy, :crop_id, :is_text_detected)

class Array
  def sum
    inject(0.0) { |result, el| result + el }
  end

  def mean
    length.zero? ? 0 : sum / length
  end
end

class String
  def intersection(other)
    str = self.dup
    other.split(//).inject(0) do |sum, char|
      sum += 1 if str.sub!(char,'')
      sum
    end
  end
end

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

def parse_estimated_bibs(json)
  return [] if json.nil?
  json['bib']['regions'].map do |r|
    if r['rbns'].nil?
      region = Region.new
      region.x1 = r['x1']
      region.x2 = r['x2']
      region.y1 = r['y1']
      region.y2 = r['y2']
      region.accuracy = r['accuracy']
      region.is_text_detected = r['is_text_detected']
      region.crop_id = r['crop_idx']
      return [region]
    end
    r['rbns'].map do |rbn|
      region = Region.new
      region.x1 = r['x1']
      region.y1 = r['y1']
      region.x2 = r['x2']
      region.y2 = r['y2']
      region.accuracy = r['accuracy']
      region.is_text_detected = r['is_text_detected']
      region.crop_id = r['crop_idx']
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

def ocr_performance(job_id, image_id, ground_truths, estimated_bibs)
  # Add total number of GT bibs
  total_bibs = ground_truths.length
  # Pool of bibs in this image
  gt_rbns = ground_truths.map(&:rbn)
  estimated_rbns = estimated_bibs.map(&:rbn).compact.reject(&:empty?)
  false_positives = estimated_rbns - gt_rbns
  false_negatives = gt_rbns - estimated_rbns
  true_positives = gt_rbns & estimated_rbns
  # Mean max character match rate
  mean_max_character_match_rate = estimated_rbns.map do |est_rbn|    
    gt_rbns.map do |gt_rbn|
      # Calculate chararcter match rate
      gt_rbn.intersection(est_rbn) / est_rbn.length.to_f
    end.max
  end.mean
  {
    job_id: job_id,
    image_id: image_id,
    mean_max_character_match_rate: mean_max_character_match_rate,
    true_positives: true_positives.join(','),
    false_negatives: false_negatives.join(','),
    false_positives: false_positives.join(','),
    true_positives_rate: true_positives.length.to_f / total_bibs,
    false_negatives_rate: false_negatives.length.to_f / total_bibs,
    false_positives_rate: false_positives.length.to_f / total_bibs
  }
end

def runtime_performance(job_id, image_id, json_for_image)
  if json_for_image.nil?
    {
      job_id: job_id,
      image_id: image_id
    }
  else
    data_hash = { person: nil }.merge(json_for_image['stats']['runtime'])
    data_hash = Hash[ data_hash.sort_by { |key, val| key.to_s } ]
    {
      job_id: job_id,
      image_id: image_id
    }.merge(data_hash)
  end
end

def txt_det_performance(job_id, image_id, estimated_bibs)
  estimated_bibs.map do |bib|
    {
      job_id: job_id,
      image_id: image_id,
      crop_id: bib.crop_id,
      model_performance: bib.is_text_detected ? 1 : 0
    }
  end
end

def bib_det_performance(job_id, image_id, ground_truths, estimated_bibs)
  p   = precision(ground_truths, estimated_bibs)
  r   = recall(ground_truths, estimated_bibs)
  f   = f_score(p, r)
  ebl = estimated_bibs.length
  gtl = ground_truths.length
  mp  = (ebl > gtl ? 1 : ebl.to_f / gtl)
  puts "#{ebl}, #{gtl}, #{mp}"
  mc  = estimated_bibs.map(&:accuracy).mean
  {
    job_id: job_id,
    image_id: image_id,
    num_gt_bibs: gtl,
    num_est_bibs: ebl,
    model_performance: mp,
    mean_confidence: mc,
    precision: p,
    recall: r,
    f_score: f
  }
end

def main
  ground_truth_json_dir = ARGV[0]
  estimated_bib_json_dir = ARGV[1]
  out_dir = ARGV[2]
  job_id = ARGV[3]

  raise 'No ground truth JSON directory provided' if ground_truth_json_dir.nil?
  raise 'No bib detect "results" JSON file provided' if estimated_bib_json_dir.nil?
  raise 'No output directory provided' if out_dir.nil?
  raise 'No job id provided' if job_id.nil?

  csv_files = {
    bib_det_performance: [],
    txt_det_performance: [],
    ocr_performance: [],
    runtime_performance: []
  }

  Dir["#{ground_truth_json_dir}/*.json"].each do |gt_file|
    # Find respective output
    image_id = File.basename(gt_file, '.jpg.json')
    results_file = "#{estimated_bib_json_dir}/results.json"
    json_for_image = JSON.parse(File.read(results_file))[image_id]
    estimated_bibs = parse_estimated_bibs(json_for_image)
    ground_truths = parse_ground_truth_bibs(gt_file)
    csv_files[:bib_det_performance] << bib_det_performance(job_id,image_id, ground_truths, estimated_bibs)
    csv_files[:txt_det_performance] += txt_det_performance(job_id, image_id, estimated_bibs)
    csv_files[:ocr_performance] << ocr_performance(job_id, image_id, ground_truths, estimated_bibs)
    csv_files[:runtime_performance] << runtime_performance(job_id, image_id, json_for_image)
  end

  FileUtils.mkdir_p(out_dir) unless Dir.exist?(out_dir)
  
  csv_files.each do |key, rows|
    csv_file = "#{out_dir}/#{key}.csv"
    csv = CSV.new(File.open(csv_file, 'wb'), headers: rows[0].keys, write_headers: true)
    rows.map(&:values).each do |row|
      csv << row
    end
  end
end

main
