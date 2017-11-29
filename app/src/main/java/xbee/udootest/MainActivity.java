package xbee.udootest;

import android.app.Activity;
import android.content.Context;
import android.hardware.usb.UsbManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import me.palazzetti.adktoolkit.AdkManager;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Attribute;
import weka.core.Debug;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import wlsvm.WLSVM;






public class MainActivity extends Activity{

//	private static final String TAG = "UDOO_AndroidADKFULL";	 
    private File root = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);

    private AdkManager mAdkManager;

    private ToggleButton buttonLED;
    private TextView distance;
    private TextView pulse;
    private TextView position;

    private AdkReadTask mAdkReadTask;

    // Part 1
    private TextView accuracyView;
    private TextView confusionView;

    // Part 2
    private EditText sepalLengthView;
    private EditText sepalWidthView;
    private EditText petalLengthView;
    private EditText petalWidthView;
    private Button trainButton;
    private Button classButton;
    private TextView svmOutView;

    // Part 3
    private ToggleButton collectOnOffButton;
    private ToggleButton collectStressRestButton;
    private Button trainStressButton;
    private ToggleButton testOnOffButton;
    private TextView outputStress;

    private boolean collect = false;
    private boolean isStress = false;
    private boolean testStress = false;

    // For Data Collection
    int BUFFER_SIZE = 10;
    private FloatBuffer pulseBuffer = FloatBuffer.allocate(BUFFER_SIZE);
    private FloatBuffer oxygenBuffer = FloatBuffer.allocate(BUFFER_SIZE);
    private FloatBuffer positionBuffer = FloatBuffer.allocate(BUFFER_SIZE);
    ArrayList<StressData> dataCollected = new ArrayList<StressData>();

    // For Stress SVM
    Classifier svmStressCls = null;
    Instances stressTestSet = null;
    FastVector fvStressWekaAttributes = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mAdkManager = new AdkManager((UsbManager) getSystemService(Context.USB_SERVICE));

//		register a BroadcastReceiver to catch UsbManager.ACTION_USB_ACCESSORY_DETACHED action
        registerReceiver(mAdkManager.getUsbReceiver(), mAdkManager.getDetachedFilter());

        buttonLED = (ToggleButton) findViewById(R.id.toggleButtonLED);
        distance  = (TextView) findViewById(R.id.textView_distance);
        pulse  = (TextView) findViewById(R.id.textView_pulse);
        position  = (TextView) findViewById(R.id.textView_position);

        // Part 1
        accuracyView = (TextView) findViewById(R.id.accuracy);
        confusionView = (TextView) findViewById(R.id.confusion);

        // Part 2
        sepalLengthView = (EditText) findViewById(R.id.sepal_length);
        sepalWidthView = (EditText) findViewById(R.id.sepal_width);
        petalLengthView = (EditText) findViewById(R.id.petal_length);
        petalWidthView = (EditText) findViewById(R.id.petal_width);
        trainButton = (Button) findViewById(R.id.train_button);
        classButton = (Button) findViewById(R.id.classify_button);
        svmOutView = (TextView) findViewById(R.id.class_output);

        // Part 3
        collectOnOffButton = (ToggleButton) findViewById(R.id.collect_on_off);
        collectStressRestButton = (ToggleButton) findViewById(R.id.collect_stress_rest);
        trainStressButton = (Button) findViewById(R.id.train_stress_button);
        testOnOffButton = (ToggleButton) findViewById(R.id.stress_test_on_off);
        outputStress = (TextView) findViewById(R.id.stress_test_output);

        BufferedReader inputTrain = null;
        BufferedReader inputTest = null;


        // File read
        InputStream train = getResources().openRawResource(R.raw.iris_train);
        inputTrain = new BufferedReader(new InputStreamReader(train));

        InputStream test = getResources().openRawResource(R.raw.iris_test);
        inputTest =  new BufferedReader(new InputStreamReader(test));


        // Read input into instance
        Instances trainData = null;
        Instances testData = null;


        // Task 1
        try {
            trainData = new Instances(inputTrain);
            testData = new Instances(inputTest);
        } catch (IOException e) {
            e.printStackTrace();
        }


        trainData.setClassIndex(trainData.numAttributes() - 1);
        Classifier ibk = new IBk();

        // Build Classified
        try {
            ibk.buildClassifier(trainData);
        } catch (Exception e) {
            e.printStackTrace();
        }

        testData.setClassIndex(testData.numAttributes() - 1);
        double count = 0;
        for (int i = 0; i < testData.numInstances(); i++) {
            double pred = 0;
            try {
                pred = ibk.classifyInstance(testData.instance(i));
            } catch (Exception e) {
                e.printStackTrace();
            }
            double act = testData.instance(i).classValue();

            if(pred == act){
                count++;
            }
        }

        double accuracy = count / testData.numInstances();
        String accuracyS = Double.toString(accuracy);
        Log.i("KNN Accuracy", accuracyS);
        accuracyView.setText(accuracyS);

        // Evaluation
        Evaluation eval = null;
        try {
            eval = new Evaluation(testData);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // Confusion matrix
        Debug.Random rand = new Debug.Random(1);
        try {
            eval.crossValidateModel(ibk, testData, 10, rand);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // print confusion matrix
        try {
            Log.i("KNN Confusion Matrix", eval.toMatrixString());
            confusionView.setText(eval.toMatrixString());
        } catch (Exception e) {
            e.printStackTrace();
        }


        // Part 2
        trainButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                BufferedReader inputTrain = null;
                BufferedReader inputTest = null;

                Toast.makeText(MainActivity.this, "Training", Toast.LENGTH_LONG).show();
                // File read
                InputStream train = getResources().openRawResource(R.raw.iris_train);
                inputTrain = new BufferedReader(new InputStreamReader(train));

                InputStream test = getResources().openRawResource(R.raw.iris_test);
                inputTest =  new BufferedReader(new InputStreamReader(test));


                // Read input into instance
                Instances trainData = null;
                Instances testData = null;

                try {
                    trainData = new Instances(inputTrain);
                    testData = new Instances(inputTest);
                } catch (IOException e) {
                    e.printStackTrace();
                }


                trainData.setClassIndex(trainData.numAttributes() - 1);
                WLSVM svmCls = new WLSVM();

                // Build Classified
                try {
                    svmCls.buildClassifier(trainData);
                } catch (Exception e) {
                    e.printStackTrace();
                }

                ObjectOutputStream oos = null;
                try {
                    oos = new ObjectOutputStream(
                            new FileOutputStream(
                                    new File(root, "svmModel.model")));
                    oos.writeObject(svmCls);
                    oos.flush();
                    oos.close();
                    Toast.makeText(MainActivity.this, "Training Complete. Saved to Downloads/svmModel.model", Toast.LENGTH_LONG).show();
                } catch (IOException e) {
                    Toast.makeText(MainActivity.this, "Train Fail", Toast.LENGTH_LONG).show();
                    e.printStackTrace();
                }

            }
        });


        classButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String sepalLengthS = sepalLengthView.getText().toString();
                Double sepalLengthD = Double.parseDouble(sepalLengthS);

                String sepalWidthS = sepalWidthView.getText().toString();
                Double sepalWidthD = Double.parseDouble(sepalWidthS);

                String petalLengthS = petalLengthView.getText().toString();
                Double petalLengthD = Double.parseDouble(petalLengthS);

                String petalWidthS = petalWidthView.getText().toString();
                Double petalWidthD = Double.parseDouble(petalWidthS);

                Classifier svmCls = null;
                try {
                    ObjectInputStream ois = new ObjectInputStream(
                            new FileInputStream(
                                    new File(root, "svmModel.model")));
                    svmCls = (Classifier) ois.readObject();
                    ois.close();
                    Toast.makeText(MainActivity.this, "Model Loaded Successfully", Toast.LENGTH_SHORT).show();
                } catch (IOException e){
                    Toast.makeText(MainActivity.this, "Failed to Load Saved Model", Toast.LENGTH_SHORT).show();
                    e.printStackTrace();
                } catch (ClassNotFoundException e){
                    Toast.makeText(MainActivity.this, "Failed to Load Saved Model", Toast.LENGTH_SHORT).show();
                    e.printStackTrace();
                }

                // Make Attribute
                Attribute Attribute1 = new Attribute("sepallength");
                Attribute Attribute2 = new Attribute("sepalwidth");
                Attribute Attribute3 = new Attribute("petallength");
                Attribute Attribute4 = new Attribute("petalwidth");

                // Declare the class attribute along with its values (nominal)
                FastVector fvClassVal = new FastVector(3);
                fvClassVal.addElement("Iris-setosa");
                fvClassVal.addElement("Iris-versicolor");
                fvClassVal.addElement("Iris-virginica");
                Attribute ClassAttribute = new Attribute("class", fvClassVal);

                // Declare the feature vector template
                FastVector fvWekaAttributes = new FastVector(5);
                fvWekaAttributes.addElement(Attribute1);
                fvWekaAttributes.addElement(Attribute2);
                fvWekaAttributes.addElement(Attribute3);
                fvWekaAttributes.addElement(Attribute4);
                fvWekaAttributes.addElement(ClassAttribute);


                Instances testingSet = new Instances("TestingInstance", fvWekaAttributes, 1);

                // Setting the column containing class labels:
                testingSet.setClassIndex(testingSet.numAttributes() - 1);

                // Create and fill an instance, and add it to the testingSet
                Instance iExample = new Instance(testingSet.numAttributes());

                iExample.setValue((Attribute)fvWekaAttributes.elementAt(0), sepalLengthD);
                iExample.setValue((Attribute)fvWekaAttributes.elementAt(1), sepalWidthD);
                iExample.setValue((Attribute)fvWekaAttributes.elementAt(2), petalLengthD);
                iExample.setValue((Attribute)fvWekaAttributes.elementAt(3), petalWidthD);
                iExample.setValue((Attribute)fvWekaAttributes.elementAt(4),
                        "Iris-setosa"); // dummy

                // add the instance
                testingSet.add(iExample);

                double predSvm = -1;
                try {
                    predSvm = svmCls.classifyInstance(testingSet.instance(0));
                } catch (Exception e) {
                    e.printStackTrace();
                }

                String output = "";

                if(predSvm == 0){
                    output = "Iris-setosa";
                } else if(predSvm == 1){
                    output = "Iris-versicolor";
                } else if (predSvm == 2){
                    output = "Iris-virginica";
                } else {
                    output = "Error";
                    Toast.makeText(MainActivity.this, "Error!", Toast.LENGTH_SHORT).show();
                }

                svmOutView.setText(output);
            }
        });


        // Part 3
        collectOnOffButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(collect){
                    collect = false;
                    Toast.makeText(MainActivity.this, "Collection Stopped", Toast.LENGTH_SHORT).show();

                    // if ArrayList of Data is not empty save the file into StressData.arff
                    if(!dataCollected.isEmpty()) {
                        // Make Attribute
                        Attribute Attribute1 = new Attribute("pulse");
                        Attribute Attribute2 = new Attribute("oxygen");
                        Attribute Attribute3 = new Attribute("position");

                        // Declare the class attribute along with its values (nominal)
                        FastVector fvClassVal = new FastVector(2);
                        fvClassVal.addElement("rest");
                        fvClassVal.addElement("stress");
                        Attribute ClassAttribute = new Attribute("class", fvClassVal);

                        // Declare the feature vector template
                        FastVector fvWekaAttributes = new FastVector(4);
                        fvWekaAttributes.addElement(Attribute1);
                        fvWekaAttributes.addElement(Attribute2);
                        fvWekaAttributes.addElement(Attribute3);
                        fvWekaAttributes.addElement(ClassAttribute);


                        Instances trainStress = new Instances("TrainingInstances", fvWekaAttributes, 50);

                        // Setting the column containing class labels:
                        trainStress.setClassIndex(trainStress.numAttributes() - 1);

                        // Create and fill an instance, and add it to the testingSet
                        // only save 50 samples
                        int count = 0;
                        for(StressData d : dataCollected) {
                            count = count + 1;
                            Instance iExample = new Instance(trainStress.numAttributes());

                            iExample.setValue((Attribute) fvWekaAttributes.elementAt(0), d.getPulseAvg());
                            iExample.setValue((Attribute) fvWekaAttributes.elementAt(1), d.getOxygenAvg());
                            iExample.setValue((Attribute) fvWekaAttributes.elementAt(2), d.getPositionAvg());

                            String stress = "";
                            if(d.getIsStress()){
                                stress = "stress";
                            } else {
                                stress = "rest";
                            }

                            iExample.setValue((Attribute) fvWekaAttributes.elementAt(3), stress);

                            // add the instance
                            trainStress.add(iExample);

                            if(count == 50){
                                break;
                            }
                        }

                        // Save to stressData.arff
                        ArffSaver saver = new ArffSaver();
                        saver.setInstances(trainStress);
                        try {
                            saver.setFile(new File(root, "stressData.arff"));
                            saver.setDestination(new File(root, "stressData.arff"));
                            saver.writeBatch();
                            Toast.makeText(MainActivity.this, "Saved to Downloads/stressData.arff", Toast.LENGTH_LONG).show();

                        } catch (IOException e) {
                            e.printStackTrace();
                        }

                    }


                } else {
                    collect = true;
                    Toast.makeText(MainActivity.this, "Collection Started", Toast.LENGTH_SHORT).show();
                }
            }
        });

        collectStressRestButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                isStress = !isStress;
            }
        });


        trainStressButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                BufferedReader inputTrain = null;

                Toast.makeText(MainActivity.this, "Training", Toast.LENGTH_LONG).show();
                // File read
                File train = new File(root, "stressData.arff");
                try {
                    inputTrain = new BufferedReader(new FileReader(train));
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }


                // Read input into instance
                Instances trainData = null;

                try {
                    trainData = new Instances(inputTrain);
                } catch (IOException e) {
                    e.printStackTrace();
                }


                trainData.setClassIndex(trainData.numAttributes() - 1);
                WLSVM svmCls = new WLSVM();

                // Build Classified
                try {
                    svmCls.buildClassifier(trainData);
                } catch (Exception e) {
                    e.printStackTrace();
                }

                ObjectOutputStream oos = null;
                try {
                    oos = new ObjectOutputStream(
                            new FileOutputStream(
                                    new File(root, "svmStressModel.model")));
                    oos.writeObject(svmCls);
                    oos.flush();
                    oos.close();
                    Toast.makeText(MainActivity.this, "Training Complete. Saved to Downloads/svmStressModel.model", Toast.LENGTH_LONG).show();
                } catch (IOException e) {
                    Toast.makeText(MainActivity.this, "Train Fail", Toast.LENGTH_LONG).show();
                    e.printStackTrace();
                }

            }

        });

        testOnOffButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (testStress) {
                    testStress = false;
                } else {
                    testStress = true;
                    try {
                        ObjectInputStream ois = new ObjectInputStream(
                                new FileInputStream(
                                        new File(root, "svmStressModel.model")));
                        svmStressCls = (Classifier) ois.readObject();
                        ois.close();
                        Toast.makeText(MainActivity.this, "Model Loaded Successfully", Toast.LENGTH_SHORT).show();
                    } catch (IOException e) {
                        Toast.makeText(MainActivity.this, "Failed to Load Saved Model", Toast.LENGTH_SHORT).show();
                        e.printStackTrace();
                    } catch (ClassNotFoundException e) {
                        Toast.makeText(MainActivity.this, "Failed to Load Saved Model", Toast.LENGTH_SHORT).show();
                        e.printStackTrace();
                    }

                    // Make Attribute
                    Attribute Attribute1 = new Attribute("pulse");
                    Attribute Attribute2 = new Attribute("oxygen");
                    Attribute Attribute3 = new Attribute("position");

                    // Declare the class attribute along with its values (nominal)
                    FastVector fvClassVal = new FastVector(2);
                    fvClassVal.addElement("rest");
                    fvClassVal.addElement("stress");
                    Attribute ClassAttribute = new Attribute("class", fvClassVal);

                    // Declare the feature vector template
                    fvStressWekaAttributes = new FastVector(4);
                    fvStressWekaAttributes.addElement(Attribute1);
                    fvStressWekaAttributes.addElement(Attribute2);
                    fvStressWekaAttributes.addElement(Attribute3);
                    fvStressWekaAttributes.addElement(ClassAttribute);

                    stressTestSet = new Instances("TestingInstance", fvStressWekaAttributes, 1);
                    stressTestSet.setClassIndex(stressTestSet.numAttributes() - 1);

                }
            }
        });

    }

    @Override
    public void onResume() {
        super.onResume();
        mAdkManager.open();

        mAdkReadTask = new AdkReadTask();
        mAdkReadTask.execute();
    }

    @Override
    public void onPause() {
        super.onPause();
        mAdkManager.close();

        mAdkReadTask.pause();
        mAdkReadTask = null;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        unregisterReceiver(mAdkManager.getUsbReceiver());
    }

    // ToggleButton method - send message to SAM3X
    public void blinkLED(View v){
        if (buttonLED.isChecked()) {
            // writeSerial() allows you to write a single char or a String object.
            mAdkManager.writeSerial("1");
        } else {
            mAdkManager.writeSerial("0");
        }
    }

    /*
     * We put the readSerial() method in an AsyncTask to run the
     * continuous read task out of the UI main thread
     */
    private class AdkReadTask extends AsyncTask<Void, String, Void> {

        private boolean running = true;

        public void pause(){
            running = false;
        }

        protected Void doInBackground(Void... params) {
//	    	Log.i("ADK demo bi", "start adkreadtask");
            while(running) {
                //TODO: UNCOMMENT THIS
                // publishProgress(mAdkManager.readSerial()) ;

                // Temporary Debug Random Values
                // Start Comment
                final String alphabet = "0123456789ABCDE";
                final int N = alphabet.length();

                Random r = new Random();

                StringBuilder s = new StringBuilder();
                for (int i = 0; i < 3; i++) {
                    s.append(alphabet.charAt(r.nextInt(N)));
                }

                publishProgress(s.toString());


                // Sleep for 1000 ms
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                // Comment till here
            }
            return null;
        }

        protected void onProgressUpdate(String... progress) {

            float pulseRate = (int) progress[0].charAt(0);
            float oxygenLvl = (int) progress[0].charAt(1);
            float pos = (int) progress[0].charAt(2);
            int max = 255;

            if (pulseRate > max) pulseRate = max;
            if (oxygenLvl > max) oxygenLvl = max;
            if (pos > max) pos = max;

//            DecimalFormat df = new DecimalFormat("#.#");

            String pulseRateS = Float.toString(pulseRate) + " (bpm)";
            distance.setText(pulseRateS);

            String oxygenLvlS = Float.toString(oxygenLvl) + " (pct)";
            pulse.setText(oxygenLvlS);

            String posS = Float.toString(pos) + "";
            position.setText(posS);

            if(collect){
                pulseBuffer.put(pulseRate);
                oxygenBuffer.put(oxygenLvl);
                positionBuffer.put(pos);

                if (!pulseBuffer.hasRemaining()
                        && !oxygenBuffer.hasRemaining()
                        && !positionBuffer.hasRemaining()) {
                    // Find average
                    double pulseSum = 0;
                    double oxygenSum = 0;
                    double positionSum = 0;

                    for (int i = 0; i < 10; i++) {
                        pulseSum = pulseSum + pulseBuffer.get(i);
                        oxygenSum = oxygenSum + oxygenBuffer.get(i);
                        positionSum = positionSum + positionBuffer.get(i);
                    }

                    // clear buffer
                    pulseBuffer.clear();
                    oxygenBuffer.clear();
                    positionBuffer.clear();

                    double pulseAvg = pulseSum / 10.0;
                    double oxygenAvg = pulseSum / 10.0;
                    double positionAvg = positionSum / 10.0;

                    // add the data to the array list
                    dataCollected.add(new StressData(pulseAvg, oxygenAvg, positionAvg, isStress));

                }
            }

            if(testStress) {
                pulseBuffer.put(pulseRate);
                oxygenBuffer.put(oxygenLvl);
                positionBuffer.put(pos);

                if (!pulseBuffer.hasRemaining()
                        && !oxygenBuffer.hasRemaining()
                        && !positionBuffer.hasRemaining()) {
                    // Find average
                    double pulseSum = 0;
                    double oxygenSum = 0;
                    double positionSum = 0;

                    for (int i = 0; i < 10; i++) {
                        pulseSum = pulseSum + pulseBuffer.get(i);
                        oxygenSum = oxygenSum + oxygenBuffer.get(i);
                        positionSum = positionSum + positionBuffer.get(i);
                    }

                    // clear buffer
                    pulseBuffer.clear();
                    oxygenBuffer.clear();
                    positionBuffer.clear();

                    double pulseAvg = pulseSum / 10.0;
                    double oxygenAvg = pulseSum / 10.0;
                    double positionAvg = positionSum / 10.0;

                    // Build iExample
                    Instance iExample = new Instance(stressTestSet.numAttributes());

                    iExample.setValue((Attribute) fvStressWekaAttributes.elementAt(0), pulseAvg);
                    iExample.setValue((Attribute) fvStressWekaAttributes.elementAt(1), oxygenAvg);
                    iExample.setValue((Attribute) fvStressWekaAttributes.elementAt(2), positionAvg);
                    iExample.setValue((Attribute) fvStressWekaAttributes.elementAt(3), "rest");

                    stressTestSet.add(iExample);

                    double predSvm = -1;
                    try {
                        predSvm = svmStressCls.classifyInstance(stressTestSet.instance(0));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }

                    String output = "";

                    if (predSvm == 0) {
                        output = "rest";
                    } else if (predSvm == 1) {
                        output = "stress";
                    } else {
                        output = "Error";
                        Toast.makeText(MainActivity.this, "Error!", Toast.LENGTH_SHORT).show();
                    }

                    outputStress.setText(output);


                }

            }
        }



    }
}


/*
StressData class to store our data into an ArrayList
 */

class StressData{
    public double pulseAvg;
    public double oxygenAvg;
    public double positionAvg;
    public boolean isStress;

    public StressData(double pulseAvg,
                      double oxygenAvg,
                      double positionAvg,
                      boolean isStress){
        this.pulseAvg = pulseAvg;
        this.oxygenAvg = oxygenAvg;
        this.positionAvg = positionAvg;
        this.isStress = isStress;
    }

    public double getPulseAvg(){
        return pulseAvg;
    }

    public double getOxygenAvg(){
        return oxygenAvg;
    }

    public double getPositionAvg(){
        return positionAvg;
    }

    public boolean getIsStress(){
        return isStress;
    }
}



