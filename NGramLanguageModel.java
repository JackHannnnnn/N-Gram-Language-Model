import java.util.*;
import java.util.regex.Pattern;
import java.io.*;
import java.text.SimpleDateFormat;


public class NGramLanguageModel {

	public static class WordsKey {
		private String[] words;
		public WordsKey() { this(1); }
		public WordsKey(int n) {
			words = new String[n];
		}
		public WordsKey(String[] words) {
			this.words = words;
		}
		
		public String[] getWords() { return words; }
		
		@Override
		public boolean equals(Object obj) {
			if (obj == null) return false;
			if (!(obj instanceof WordsKey))
				return false;
			WordsKey other = (WordsKey) obj;
			return Arrays.equals(words, other.getWords());
		}
		
		@Override
		public int hashCode() {
			return Arrays.hashCode(words);
		}
	}
	
	
	private int n;
	private double KSmoothing;
	private double[] linearInterpoParams = null;
	private int lowFrequencyThreshold;
	private int unkThreshold;
	private int trainLineCount;
	private Map<WordsKey, Long>[] countContainer;
	private Map<WordsKey, Double>[] modelParams;
	private TreeSet<String> vocabulary;
	private Map<String, String> mappingList;

	
	public NGramLanguageModel(int N) {
		this(N, 0, null, 5, 6);
	}
	
	public NGramLanguageModel(int N, int K) {
		this(N, K, null, 5, 6);
	}
	
	public NGramLanguageModel(int N, double[] linearInterpoParams) {
		this(N, 0, linearInterpoParams, 5, 6);
	}
	
	public NGramLanguageModel(int N, double KSmoothing, double[] linearInterpoParams, int lowFre, int unkThre) {
		if (N < 1 || KSmoothing < 0)
			throw new IllegalArgumentException("N for N-gram must be greater than 0 and K for add-K must be non-negative!");
		this.n = N;
		this.KSmoothing = KSmoothing;
		
		if (!checkLinearInterpoParams(linearInterpoParams))
			throw new IllegalArgumentException("Sum of lambdas must be 1 and each lambda must be non-negative!");
		this.linearInterpoParams = linearInterpoParams;
		
		if (lowFre <= 0 || unkThre <= 0)
			throw new IllegalArgumentException("The low frequency threshold and the unk threshold must be greater than 0!");
		this.lowFrequencyThreshold = lowFre;
		this.unkThreshold = unkThre;
		
		this.countContainer = new HashMap[n];
		this.modelParams = new HashMap[n];
		this.vocabulary = new TreeSet<>();
		this.mappingList = new HashMap<>();
		this.trainLineCount = 0;
		
		for (int i = 0; i < n; i++) {
			countContainer[i] = new HashMap<>();
			modelParams[i] = new HashMap<>();
		}
	}
	
	public void train(String filename) throws IOException {
		Scanner data = new Scanner(new File(System.getProperty("user.dir") + "/" + filename));
		if (linearInterpoParams != null) {
			while (data.hasNextLine()) {
				String[] words = (data.nextLine() + " <STOP>").split(" ");
				for (int k = 0; k < words.length; k++) {
					for (int j = 1; j <= this.n; j++) {
						updateCount(words, k, j);
					}	
				}
				trainLineCount++;
			}
		} else {
			while (data.hasNextLine()) {
				String[] words = (data.nextLine() + " <STOP>").split(" ");
				for (int k = 0; k < words.length; k++) {
					updateCount(words, k, 1);
					if (n > 1)
						updateCount(words, k, n);
					if (n-1 > 1)
						updateCount(words, k, n-1);
				}
				trainLineCount++;
			}
		}
		data.close();
		
		preprocess(saveVocabularyCount());
		
		if (linearInterpoParams == null)
			paramEstimate(KSmoothing, n);
		else {
			for (int i = 1; i <= n; i++)
				paramEstimate(KSmoothing, i);
			for (WordsKey each : modelParams[n-1].keySet()) {
				double linearInterpoEstimate = 0;
				for (int i = 0; i < n; i++) {
					WordsKey wk = new WordsKey(Arrays.copyOfRange(each.getWords(), i, n));
					double param = modelParams[n-i-1].get(wk) == null ? (1.0 / vocabulary.size()) : modelParams[n-i-1].get(wk);
					linearInterpoEstimate += linearInterpoParams[n-i-1] * param;
				}
				modelParams[n-1].put(each, linearInterpoEstimate);
			}
		}
		System.out.println("Training is done!");
		
	}
	
	
	private void preprocess(SortedSet<Map.Entry<WordsKey, Long>> sortedVoca) throws IOException {
		String[] patterns = { 
				"^\\d\\d$", 
				"^\\d\\d\\d\\d*", 
				"^(?=.*[0-9])(?=.*[a-zA-Z])([a-zA-Z0-9_-]+)$",
				"^(?=.*[0-9])(?=.*[-])([0-9-]+)$",
				"^(?=.*[0-9])(?=.*[/])([0-9/]+)$",
				"^(?=.*[0-9])(?=.*[,])([0-9,]+)$",
				"^(?=.*[0-9])(?=.*[.])([0-9.]+)$",
				"^[0-9]{1}|[0-9]{3}|[0-9]{5,}$",
				"^[A-Z]+$",
				"^[A-Z].$",
				"^[A-Z][a-z]*$",
				"^[a-z]+$" 
		};
		String[] newClasses = { 
				"twoDigitNum",
				"fourDigitNum",
				"containsDigitAndAlpha",
				"containsDigitAndDash",
				"containsDigitAndSlash",
				"containsDigitAndComma",
				"containsDigitAndPeriod",
				"otherNum",
				"allCaps",
				"capPeriod",
				"initCap",
				"lowerCase"
		};
		
		Pattern[] patternObj = new Pattern[patterns.length];
		for (int i = 0; i < patternObj.length; i++)
			patternObj[i] = Pattern.compile(patterns[i]);
		
		BufferedWriter br = new BufferedWriter(new FileWriter(System.getProperty("user.dir") + "/mappingList.txt"));
		for (Map.Entry<WordsKey, Long> each : sortedVoca) {
			String word = each.getKey().getWords()[0];
			if (each.getValue() <= lowFrequencyThreshold) {
				boolean skip = false;
				for (int j = 0; j < patternObj.length; j++) {
					if (patternObj[j].matcher(word).matches()) {
						mappingList.put(word, newClasses[j]);
						br.write(word + "\t\t=>\t" + newClasses[j]);
						br.newLine();
						skip = true;
					}
				}
				if (!skip) {
					mappingList.put(word, "<OTHER>");
					br.write(word + "\t\t=>\t" + "<OTHER>");
					br.newLine();
				}
			}
		}
		br.close();
		System.out.println("Named Entity Recognition is done and the mapping list is saved!");
		
		Iterator<Map.Entry<WordsKey, Long>> iterator = countContainer[0].entrySet().iterator();
		Map<WordsKey, Long> update = new HashMap<>();
		while (iterator.hasNext()) {
			Map.Entry<WordsKey, Long> each = iterator.next();
			String keyString = each.getKey().getWords()[0];
			if (mappingList.containsKey(keyString)) {
				WordsKey dest = new WordsKey(new String[] { mappingList.get(keyString) });
				long ori = update.get(dest) == null ? 0 : update.get(dest);
				update.put(dest, each.getValue() + ori);
				iterator.remove();
			}
		}
		countContainer[0].putAll(update);
		
		
		
		TreeSet<String> wordsConverted = new TreeSet<>();
		iterator = countContainer[0].entrySet().iterator();
		update = new HashMap<>();
		WordsKey unk = new WordsKey(new String[] { "<UNK>" });
		update.put(unk, 0l);
		vocabulary.add("<UNK>");
		int count = 0;
		while (iterator.hasNext()) {
			Map.Entry<WordsKey, Long> each = iterator.next();
			if (each.getValue() <= unkThreshold) {
				update.put(unk, each.getValue() + update.get(unk));
				wordsConverted.add(each.getKey().getWords()[0]);
				iterator.remove();
				count++;
			} else {
				vocabulary.add(each.getKey().getWords()[0]);
			}
		}
		countContainer[0].putAll(update);
		System.out.println(String.format("%d words have been identified as Named Entity!", count));
		
		
		for (int j = 1; j < n; j++) {
			iterator = countContainer[j].entrySet().iterator();
			Map<WordsKey, Long> temp = new HashMap<>();
			while(iterator.hasNext()) {
				Map.Entry<WordsKey, Long> each = iterator.next();
				WordsKey newKey = new WordsKey(j+1);
				for (int m = 0; m < j+1; m++) {
					String word = each.getKey().getWords()[m];
					
					if (mappingList.containsKey(word)) {
						each.getKey().getWords()[m] = mappingList.get(word);
						word = each.getKey().getWords()[m];
					}
					if (wordsConverted.contains(word)) {
						newKey.getWords()[m] = "<UNK>";
					} else 
						newKey.getWords()[m] = word;
				}
				if (!newKey.equals(each.getKey())) {
					Long addCount = each.getValue();
					iterator.remove();
					temp.put(newKey, addCount + ((temp.get(newKey) == null) ? 0 : temp.get(newKey)));
				}
			}
			countContainer[j].putAll(temp);
		}
	}
	
	
	private void paramEstimate(double K, int ngram) {
		int vocabularySize = countContainer[0].size();
		long wordTotalSize = 0;
		for (Long each : countContainer[0].values())
			wordTotalSize += each;
	
		if (ngram == 1) {
			for (WordsKey each : countContainer[0].keySet()) {
				Double estimate = (countContainer[0].get(each) + K) / (wordTotalSize + K*vocabularySize);
				modelParams[ngram-1].put(each, estimate);
			}
		} else {
			for (Map.Entry<WordsKey, Long> each : countContainer[ngram-1].entrySet()) {
				WordsKey denominatorKey = new WordsKey(Arrays.copyOfRange(each.getKey().getWords(), 0, ngram-1));
				double denominator = 0;
				if (countContainer[ngram-2].get(denominatorKey) == null) {
					denominator = trainLineCount + K*vocabularySize;
				}
				else
					denominator = countContainer[ngram-2].get(denominatorKey) + K*vocabularySize;
				Double estimate = (each.getValue() + K) / denominator;
				modelParams[ngram-1].put(each.getKey(), estimate);
			}
		}
		
	}
	
	
	private void updateCount(String[] words, int k, int ngram) {
		WordsKey key = new WordsKey(ngram);
		if (k >= ngram - 1) {			
			for (int l = 0; l < ngram; l++)
				key.getWords()[l] = words[k-ngram+1+l];		
		} else {
			for (int l = 0; l < ngram; l++) {
				if (k-ngram+1+l < 0)
					key.getWords()[l] = "<START>";
				else 
					key.getWords()[l] = words[k-ngram+1+l];
			}
		}
		if (countContainer[ngram-1].get(key) == null)
			countContainer[ngram-1].put(key, 1l);
		else {
			countContainer[ngram-1].put(key, countContainer[ngram-1].get(key) + 1);
		}
	}
	
	private SortedSet<Map.Entry<WordsKey, Long>> saveVocabularyCount() throws IOException {
		SortedSet<Map.Entry<WordsKey, Long>> sortedVocaCount = new TreeSet<Map.Entry<WordsKey, Long>>(
			new Comparator<Map.Entry<WordsKey, Long>>() {
				@Override
				public int compare(Map.Entry<WordsKey, Long> e1, Map.Entry<WordsKey, Long> e2) {
					int res = e1.getValue().compareTo(e2.getValue());
					if (res == 0)
						return e1.getKey().equals(e2.getKey()) ? 0 : -1;
					return -res;
			}
		});
		
		sortedVocaCount.addAll(countContainer[0].entrySet());
		BufferedWriter br = new BufferedWriter(new FileWriter(System.getProperty("user.dir") + "/VocabularyCount.txt"));

		for (Map.Entry<WordsKey, Long> each : sortedVocaCount) {
			br.write(each.getKey().getWords()[0] + "\t\t\t" + each.getValue());
			br.newLine();
		}
		br.close();
		System.out.println("Original vocabulary counts are saved!");
		return sortedVocaCount;
	}
	
	public Double[] predict(String filename) throws FileNotFoundException {
		Scanner data = new Scanner(new File(System.getProperty("user.dir") + "/" + filename));
		List<Double> result = new ArrayList<>();
		double wordsTotalSize = 0;
		while (data.hasNextLine()) {
			String[] words = (data.nextLine() + " <STOP>").split(" ");
			wordsTotalSize += words.length;
			double logProb = 0;
			for (int k = 0; k < words.length; k++) {
				WordsKey wk = new WordsKey(n);
				String word = mappingList.containsKey(words[k]) ? mappingList.get(words[k]) : words[k];
				wk.getWords()[n-1] = vocabulary.contains(word) ? word : "<UNK>";
				for (int j = k-1; j >= k-n+1; j--) {
					if (j < 0)
						wk.getWords()[j-k+n-1] = "<START>";
					else {
						word = mappingList.containsKey(words[j]) ? mappingList.get(words[j]) : words[j];
						wk.getWords()[j-k+n-1] = (vocabulary.contains(word) ? word : "<UNK>");
					}
				}
				double estimate = 0;
				if (modelParams[n-1].get(wk) == null && linearInterpoParams == null) {
					WordsKey denominatorKey = new WordsKey(Arrays.copyOfRange(wk.getWords(), 0, n-1));
					double denominator = getDenominator(n-2, denominatorKey);
					estimate = KSmoothing / (denominator + KSmoothing*vocabulary.size());
					modelParams[n-1].put(wk, estimate);
				} else if (modelParams[n-1].get(wk) == null && linearInterpoParams != null) {
					for (int i = 0; i < n; i++) {
						WordsKey numeratorKey = new WordsKey(Arrays.copyOfRange(wk.getWords(), i, n));
						if (modelParams[n-i-1].get(numeratorKey) != null)
							estimate += linearInterpoParams[n-i-1] * modelParams[n-i-1].get(numeratorKey);
						else {
							WordsKey denominatorKey = new WordsKey(Arrays.copyOfRange(wk.getWords(), i, n-1));
							double denominator = getDenominator(n-i-2, denominatorKey);
							if (denominator == 0 && KSmoothing == 0)
								continue;
							estimate += linearInterpoParams[n-i-1] * (KSmoothing / (denominator + KSmoothing*vocabulary.size()));
						}
					}
					modelParams[n-1].put(wk, estimate);
				} else {
					estimate = modelParams[n-1].get(wk);
				}
				logProb += Math.log(estimate) / Math.log(2);
			}
			result.add(logProb);
		}
		data.close();
		result.add(wordsTotalSize);
		System.out.println("Prediction is done!");
		return result.toArray(new Double[result.size()]);
	}
	
	private double getDenominator(int loc, WordsKey denominatorKey) {
		double denominator;
		if (countContainer[loc].get(denominatorKey) != null)
			denominator = countContainer[loc].get(denominatorKey);
		else if (denominatorKey.getWords()[loc] == "<START>")
			denominator = trainLineCount;
		else
			denominator = 0;
		return denominator;
	}
	
	public void savePredictions(Double[] pred, String filename) throws IOException {
		BufferedWriter br = new BufferedWriter(new FileWriter(System.getProperty("user.dir") + "/" + filename));
		for (int i = 0; i < pred.length-1; i++) {
			br.write(String.valueOf(pred[i]));
			br.newLine();
		}
		br.close();
		System.out.println("Prediction results (Log probability) are saved!");
	}
	
	public double evaluate(Double[] logPreds) {
		double perplexity = 0;
		for (int i = 0; i < logPreds.length-1; i++) {
			double logProb = logPreds[i];
			perplexity += logProb;
		}
		perplexity = Math.pow(2, -(perplexity/logPreds[logPreds.length-1]));
		System.out.println("Perplexity of the prediction data set: " + perplexity);
		return perplexity;
	}
	
	private static boolean checkLinearInterpoParams(double[] params) {
		if (params == null) return true;
		double sum = 0;
		for (double lambda : params) {
			if (lambda < 0) return false;
			sum += lambda;
		}
		return sum == 1;
	}
	
	public static void main(String[] args) throws IOException {
		
		String trainFile = args[0];
		String predictFile = args[1];
		int ngram = Integer.parseInt(args[2]);
		double K = Double.parseDouble(args[3]);
		double[] linearInterpoParams;
		if (args.length > 5) {
			linearInterpoParams = new double[ngram];
			for (int i = 0; i < ngram; i++) {
				linearInterpoParams[i] = Double.parseDouble(args[i+4]);
			}
		} else
			linearInterpoParams = null;
		
		NGramLanguageModel lm;
		if (args.length > 4 + ngram) {
			int lowFreThre = Integer.parseInt(args[ngram+4]), unkThre = Integer.parseInt(args[ngram+5]);
			lm = new NGramLanguageModel(ngram, K, linearInterpoParams, lowFreThre, unkThre);
		} else
			lm = new NGramLanguageModel(ngram, K, linearInterpoParams, 5, 6);
		
		lm.train(trainFile);
		Double[] pred = lm.predict(predictFile);
		String savePred = new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date());
		lm.savePredictions(pred, "/pred-" + savePred + ".txt");
		lm.evaluate(pred);
	}
}
