import java.io.File;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.Set;

public class LogisticRgression {
	static HashMap<String, Double> weightMatrix = new HashMap<>();
	static double learningRate;
	static double lambdaParameter;
	static int number_of_iterations;
	static double w0 = 0.1;

	static int num_ham = 0;
	static int num_spam = 0;
	
	public static HashMap<String, HashMap<String, Integer>> spamFileDetails = new HashMap<>();
	public static HashMap<String, HashMap<String, Integer>> hamFileDetails = new HashMap<>();
	public static HashMap<String, Integer> spamTokens = new HashMap<>();
	public static HashMap<String, Integer> hamTokens = new HashMap<>();
	public static HashMap<String, Double> hamWordsProbability = new HashMap<>();
	public static HashMap<String, Double> spamWordsProbability = new HashMap<>();
	public static Set<String> stopWordsList = new HashSet<>();
	public static Set<String> tokens = new HashSet<>();
	static Set<String> hamSet = new HashSet<>();
	static Set<String> spamSet = new HashSet<>();
	static Set<String> allFileSet = new HashSet<>();
	static int hamDenominator =0;
	static int spamDenominator = 0;
	static int hamNumerator = 0;
	static int spamNumerator = 0;
	static double hamProbability = 0;
	static double spamProbability = 0;
	static double likelihoodHam = 0;
	static double likeliHoodSpam =0;
	
	static int getHamDenominator() {
		for (int values : hamTokens.values()) {
			hamDenominator = hamDenominator + values +1;
		}
		return hamDenominator;
	}
	
	static int getSpamDenominator() {
		for (int values : spamTokens.values()) {
			spamDenominator = spamDenominator + values +1;
		}
		return spamDenominator;
	}
	
	//get probability for each word belonging to ham or spam
	static void getProbabilityHamSpam(int hmd, int smd)
	{
		for(String t:tokens)
		{
			hamNumerator = hamTokens.getOrDefault(t, 0);
			hamNumerator += 1;
			hamProbability = (double) hamNumerator/hmd;
			hamWordsProbability.put(t, hamProbability);
			
			spamNumerator = spamTokens.getOrDefault(t, 0);
			spamNumerator += 1;
			spamProbability = (double) spamNumerator/smd;
			spamWordsProbability.put(t, spamProbability);
		}
	}
	
	private static double sigmoid(double net) {
		if (net < -100)
			return 0.0;

		else if (net > 100)
			return 1.0;
		else
			return (1.0 / (1.0 + Math.exp(-net)));
	}
	
	private static double zNetFile(String doc) {
		if (spamSet.contains(doc)) {
			double zNetSpam = w0;
			try {
				for (Entry<String, Integer> fn : spamFileDetails.get(doc).entrySet()) {
					zNetSpam += (fn.getValue()) * weightMatrix.get(fn.getKey());
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
			return sigmoid(zNetSpam);
		} else {
			double zNetHam = w0;
			try {
				for (Entry<String, Integer> fh : hamFileDetails.get(doc).entrySet()) {
					zNetHam += (fh.getValue()) * weightMatrix.get(fh.getKey());
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
			return sigmoid(zNetHam);
		}
	}

	
	//get unique words in all files
	public static void getUniqueWordsInAllFiles(File pathToDir) throws Exception {
		for (File files : pathToDir.listFiles()) {
			Scanner scanner = new Scanner(files);
			while (scanner.hasNext()) {
				String line = scanner.nextLine();
				for (String words : line.toLowerCase().trim().split(" ")) {
					words = words.replaceAll("[0-9]+", "");
					words = words.replaceAll("\\.", "");
					words = words.replaceAll("-", "");
					words = words.replaceAll("\\'", "");
					words = words.replaceAll("\\'", "");
					words = words.replaceAll("'s", "");
					words = words.replaceAll("[+^:,?;=%#&~`$!@*_)/(}{]", "");
					words = words.replaceAll("\\<.*?>", "");
					if (!words.isEmpty()) {
						tokens.add(words);
					}
				}
			}
			scanner.close();
		}
	}

	private static int getFrequencyOfWordInAFile(String doc, String t) {
		int freq = 0;
		if (spamSet.contains(doc)) {
			try {
				for (Entry<String, Integer> fs : spamFileDetails.get(doc).entrySet()) {
					if (fs.getKey().equals(t)) {
						freq = fs.getValue();
						return freq;
					}
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else if (hamSet.contains(doc)) {
			try {
				for (Entry<String, Integer> fh : hamFileDetails.get(doc).entrySet()) {
					if (fh.getKey().equals(t)) {
						freq = fh.getValue();
						return freq;
					}
				}
			} catch (Exception e) {
				e.printStackTrace();
			}

		}
		return 0;
	}

	
	public static double classifyTestFiles(File mails, int expected_output, String removeStopWords) throws Exception
	{
		if(expected_output==1)
		{
			int correctly_classifed_spam = 0;
			for (File testFile : mails.listFiles())
			{
				num_spam = num_spam +1;
				HashMap<String, Integer>testFileHm = new HashMap<>();
				Scanner sc = new Scanner(testFile);
				while(sc.hasNext())
				{
					String line = sc.nextLine();
					for(String words: line.toLowerCase().trim().split(" "))
					{
						words = words.replaceAll("[0-9]+","");
						words = words.replaceAll("\\.", "");
						words= words.replaceAll("-", "");
						words =words.replaceAll("\\'", "");
						words =words.replaceAll("\\'", "");
						words =words.replaceAll("'s", "");
						words =words.replaceAll("[+^:,?;=%#&~`$!@*_)/(}{]", "");
						words =words.replaceAll("\\<.*?>", "");
						if(testFileHm.containsKey(words))
						{
							testFileHm.put(words, testFileHm.get(words)+1);
						}
						else
						{
							testFileHm.put(words, 1);
						}
					}
				}
				
				sc.close();
				if(removeStopWords.equals("Yes"))
				{
					for (String stopWord: stopWordsList)
					{
						if(testFileHm.containsKey(stopWord))
						{
							testFileHm.remove(stopWord);
						}
					}
				}
				int res = testFile(testFileHm);
				if(res == 1)
					correctly_classifed_spam+=1;
			}
			return correctly_classifed_spam;
		}
		else
		{
			int correctly_classifed_ham=0;
			num_ham = mails.listFiles().length;
			for(File testFile: mails.listFiles())
			{
				HashMap<String, Integer>testFileHm = new HashMap<>();
				Scanner sc = new Scanner(testFile);
				while(sc.hasNext())
				{
					String line = sc.nextLine();
					for(String words: line.toLowerCase().trim().split(" "))
					{
						words = words.replaceAll("[0-9]+","");
						words = words.replaceAll("\\.", "");
						words= words.replaceAll("-", "");
						words =words.replaceAll("\\'", "");
						words =words.replaceAll("\\'", "");
						words =words.replaceAll("'s", "");
						words =words.replaceAll("[+^:,?;=%#&~`$!@*_)/(}{]", "");
						words =words.replaceAll("\\<.*?>", "");
						if(testFileHm.containsKey(words))
						{
							testFileHm.put(words, testFileHm.get(words)+1);
						}
						else
						{
							testFileHm.put(words, 1);
						}
					}
					
				}
				sc.close();
				int re = testFile(testFileHm);
				if(re==0)
				{
					correctly_classifed_ham+=1;
				}
			}
			return correctly_classifed_ham;
		}
	}

	public static double classifyTestFilesNB(File mails, int expected_output, String removeStopWords) throws Exception
	{
		if(expected_output==1)
		{
			int correctly_classifed_spam = 0;
			for (File testFile : mails.listFiles())
			{
				//num_spam = num_spam +1;
				HashMap<String, Integer>testFileHm = new HashMap<>();
				Scanner sc = new Scanner(testFile);
				while(sc.hasNext())
				{
					String line = sc.nextLine();
					for(String words: line.toLowerCase().trim().split(" "))
					{
						words = words.replaceAll("[0-9]+","");
						words = words.replaceAll("\\.", "");
						words= words.replaceAll("-", "");
						words =words.replaceAll("\\'", "");
						words =words.replaceAll("\\'", "");
						words =words.replaceAll("'s", "");
						words =words.replaceAll("[+^:,?;=%#&~`$!@*_)/(}{]", "");
						words =words.replaceAll("\\<.*?>", "");
						if(testFileHm.containsKey(words))
						{
							testFileHm.put(words, testFileHm.get(words)+1);
						}
						else
						{
							testFileHm.put(words, 1);
						}
					}
				}
				
				sc.close();
				if(removeStopWords.equals("Yes"))
				{
					for (String stopWord: stopWordsList)
					{
						if(testFileHm.containsKey(stopWord))
						{
							testFileHm.remove(stopWord);
						}
					}
				}
				int res = testFileNB(testFileHm, 1);
				if(res == 1)
					correctly_classifed_spam+=1;
			}
			return correctly_classifed_spam;
		}
		else
		{
			int correctly_classifed_ham=0;
			num_ham = mails.listFiles().length;
			for(File testFile: mails.listFiles())
			{
				HashMap<String, Integer>testFileHm = new HashMap<>();
				Scanner sc = new Scanner(testFile);
				while(sc.hasNext())
				{
					String line = sc.nextLine();
					for(String words: line.toLowerCase().trim().split(" "))
					{
						words = words.replaceAll("[0-9]+","");
						words = words.replaceAll("\\.", "");
						words= words.replaceAll("-", "");
						words =words.replaceAll("\\'", "");
						words =words.replaceAll("\\'", "");
						words =words.replaceAll("'s", "");
						words =words.replaceAll("[+^:,?;=%#&~`$!@*_)/(}{]", "");
						words =words.replaceAll("\\<.*?>", "");
						if(testFileHm.containsKey(words))
						{
							testFileHm.put(words, testFileHm.get(words)+1);
						}
						else
						{
							testFileHm.put(words, 1);
						}
					}
					
				}
				sc.close();
				int re = testFileNB(testFileHm, 0);
				if(re==0)
				{
					correctly_classifed_ham+=1;
				}
			}
			return correctly_classifed_ham;
		}
	}
	
	public static int testFile(HashMap<String, Integer> hmtest) {
		double result = 0;
		for (Entry<String, Integer> m : hmtest.entrySet()) {
			if (weightMatrix.containsKey(m.getKey())) {
				result += (m.getValue() * weightMatrix.get(m.getKey()));
			}
		}
		result = result + w0;
		if (result < 0)
			return 0;
		else
			return 1;
	}
	
	public static int testFileNB(HashMap<String, Integer> hmtest, int target) {
		double likelihood_ham = 0;
		double likelihood_spam = 0;
		double posterior_ham = 0;
		double posterior_spam = 0;
		double prior_ham = (double) num_ham / (num_ham + num_spam);
		double prior_spam = (double) num_spam / (num_ham + num_spam);
		double log_prior_ham = 0;
		double log_prior_spam = 0;
		double num1,num2;
		for (String keys : hmtest.keySet()) {
			if(!keys.isEmpty())
			{
			 num1= Math.log(hamWordsProbability.getOrDefault(keys, 1.0));
			 num2 = Math.log(spamWordsProbability.getOrDefault(keys, 1.0));
			likelihood_ham += (num1/Math.log(2));
			likelihood_spam += (num2/Math.log(2));
			}
		}
		if (prior_ham != 0)
			log_prior_ham = Math.log(prior_ham)/Math.log(2);
		if (prior_spam != 0)
			log_prior_spam = Math.log(prior_spam)/Math.log(2);
		posterior_ham = log_prior_ham + likelihood_ham;
		posterior_spam = log_prior_spam + likelihood_spam;
		if (target == 1) {
			if (posterior_spam > posterior_ham)
				return 1;
			else
				return 0;
		} else {
			if (posterior_spam < posterior_ham)
				return 0;
			else
				return 1;

		}
	}

	private static void fileDetails(File pathToFiles, int expected_output) throws Exception {
		if (expected_output == 1) {
			for (File spamFiles : pathToFiles.listFiles()) {
				HashMap<String, Integer> fileWordFreq = new HashMap<>();
				spamSet.add(spamFiles.getName());
				allFileSet.add(spamFiles.getName());
				Scanner sc = new Scanner(spamFiles);
				while (sc.hasNext()) {
					String line = sc.nextLine();
					for (String words : line.toLowerCase().trim().split(" ")) {
						words = words.replaceAll("[0-9]+", "");
						words = words.replaceAll("\\.", "");
						words = words.replaceAll("-", "");
						words = words.replaceAll("\\'", "");
						words = words.replaceAll("\\'", "");
						words = words.replaceAll("'s", "");
						words = words.replaceAll("[+^:,?;=%#&~`$!@*_)/(}{]", "");
						words = words.replaceAll("\\<.*?>", "");
						if (tokens.contains(words)) {
							if (spamTokens.containsKey(words)) {
								spamTokens.put(words, spamTokens.get(words) + 1);
							} else {
								spamTokens.put(words, 1);
							}
							if (fileWordFreq.containsKey(words)) {
								fileWordFreq.put(words, fileWordFreq.get(words) + 1);
							} else {
								fileWordFreq.put(words, 1);
							}
						}
						spamFileDetails.put(spamFiles.getName(), fileWordFreq);
					}

				}
				sc.close();
			}
		} else {
			for (File hamFiles : pathToFiles.listFiles()) {
				HashMap<String, Integer> fileWordFreq = new HashMap<>();
				hamSet.add(hamFiles.getName());
				allFileSet.add(hamFiles.getName());
				Scanner sc = new Scanner(hamFiles);
				while (sc.hasNext()) {
					String line = sc.nextLine();
					for (String words : line.toLowerCase().trim().split(" ")) {
						words = words.replaceAll("[0-9]+", "");
						words = words.replaceAll("\\.", "");
						words = words.replaceAll("-", "");
						words = words.replaceAll("\\'", "");
						words = words.replaceAll("\\'", "");
						words = words.replaceAll("'s", "");
						words = words.replaceAll("[+^:,?;=%#&~`$!@*_)/(}{]", "");
						words = words.replaceAll("\\<.*?>", "");
						if (!words.isEmpty()) {
							if (tokens.contains(words)) {
								if (hamTokens.containsKey(words)) {
									hamTokens.put(words, hamTokens.get(words) + 1);
								} else {
									hamTokens.put(words, 1);
								}
							}
						}
						if (!words.isEmpty()) {
							if (tokens.contains(words)) {
								if (fileWordFreq.containsKey(words)) {
									fileWordFreq.put(words, fileWordFreq.get(words) + 1);
								} else {
									fileWordFreq.put(words, 1);
								}
							}

						}
						hamFileDetails.put(hamFiles.getName(), fileWordFreq);
					}

				}
				sc.close();
			}
		}
	}

	public static void trainWeights(int itr) {
		int target = 0;
		// provide some random weights to begin with
		for (String t : tokens) {
			double w = 2 * Math.random() - 1;
			weightMatrix.put(t, w);
		}
		for (int i = 0; i < itr; i++) {
			for (String t : tokens) {
				double totalError = 0;
				for (String doc : allFileSet) {
					int freq = getFrequencyOfWordInAFile(doc, t);
					if (spamSet.contains(doc)) {
						target = 1;
					} else {
						target = 0;
					}
					double sigmoidOutput = zNetFile(doc);
					double E = (target - sigmoidOutput);
					totalError = totalError + freq * E;
				}
				double newWt = weightMatrix.get(t) + learningRate * (totalError - (lambdaParameter * weightMatrix.get(t)));
				weightMatrix.put(t, newWt);
			}
		}
	}
public static void main(String[]args) throws Exception
{
	if(args.length!=6)
	{
		System.out.println("Unexpected number of arguments. Exiting...");
		System.out.println(args.length);
		for(int i =0;i<6;i++)
		{
			System.out.println(args[i]);
		}
		return;
	}
	
	String train = args[0];
	String test = args[1];
	File spamPathTrain = new File(train+"/spam");
	File hamPathTrain = new File(train+"/ham");
	File spamPathTest = new File(test+"/spam");
	File hamPathTest = new File(test+"/ham");
	String removeStopWords = args[2];
	learningRate = Double.parseDouble(args[3]);
	lambdaParameter = Double.parseDouble(args[4]);
	number_of_iterations = Integer.parseInt(args[5]);
	int hmd =0;
	int smd = 0;
	
	getUniqueWordsInAllFiles(spamPathTrain);
	getUniqueWordsInAllFiles(hamPathTrain);
	if(removeStopWords.equalsIgnoreCase("Yes"))
	{
		File stopwords = new File("stopWords.txt");
		Scanner st = null;
		try
		{
			
			st = new Scanner(stopwords);
		}catch(Exception e)
		{
			e.printStackTrace();
		}
		while(st.hasNext())
		{
			String sw = st.next();
			stopWordsList.add(sw);
		}
	
		st.close();
		for(String sw : stopWordsList)
		{
			if(tokens.contains(sw))
				tokens.remove(sw);
		}
	}
	fileDetails(spamPathTrain, 1);
	fileDetails(hamPathTrain, 0);
	trainWeights(number_of_iterations);
	double sAcc = classifyTestFiles(spamPathTest, 1, removeStopWords);
	double hAcc = classifyTestFiles(hamPathTest, 0, removeStopWords);
	System.out.println("Accuracy on test data LR " + ((sAcc+hAcc)/(num_ham+num_spam))*100);
	hmd = getHamDenominator();
	smd =getSpamDenominator();
	getProbabilityHamSpam(hmd, smd);
	double sNBAcc = classifyTestFilesNB(spamPathTest,1, removeStopWords);
	double hNBAcc = classifyTestFilesNB(hamPathTest,0, removeStopWords);
	System.out.println("Accuracy on test data NB " + ((sNBAcc+hNBAcc)/(num_ham+num_spam))*100);
}
}
