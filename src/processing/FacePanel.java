package processing;

/* 
    Este painel repetidamente tira imagens do dispositivo de captura e as desenha no painel.
    Uma face é destacada com um retângulo verde, que é atualizado conforme a face se movimenta.
    
    A face destacada pode ser armazenada para reconhecimento da pessoa.

    A tarefa de detecção é realizado pelo Haar cascade classifier disponibilizado pelo JavaCV.
    Esta tarefa é executada em sua própria thread devido o processo ser mais lento. sendo assim
    a captura das imagens não é afetada pelo reconhecimento dos rostos.
 */
import java.awt.*;
import javax.swing.*;
import java.awt.image.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_objdetect.*;
import static org.bytedeco.javacpp.helper.opencv_objdetect.cvHaarDetectObjects;
import org.bytedeco.javacpp.opencv_objdetect.CvHaarClassifierCascade;
import org.bytedeco.javacpp.videoInputLib.videoInput;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter.ToIplImage;

public class FacePanel extends JPanel implements Runnable {

    private FrameGrabber grabber;

    /* Dimensões de cada imagem; o painel é do mesmo tamanho das imagens */
    private static final int WIDTH = 640;
    private static final int HEIGHT = 480;

    private static final int DELAY = 100;  // Tempo (ms) delay para desenhar o painel

    private static final int CAMERA_ID = 0;

    private static final int IM_SCALE = 4;
    private static final int SMALL_MOVE = 5;

    // tempo (ms) entre cada detecção de face
    private static final int DETECT_DELAY = 500;

    private static final int MAX_TASKS = 4;
    // Nr. máximo de tarefas que podem ficar aguardando no executor

    // Modelo cascata utilizado para detecção das faces (Disponibilizado pelo OpenCV)
    private static final String FACE_CASCADE_FNM = "haarcascade_frontalface_alt.xml";
    // "haarcascade_frontalface_alt2.xml";

    // Atributos para salvar uma imagem
    private static final String FACE_DIR = "savedFaces";
    private static final String FACE_FNM = "face";
    private static String FACE_ID;
    private static final int FACE_WIDTH = 125;
    private static final int FACE_HEIGHT = 150;

    private IplImage snapIm = null;
    private volatile boolean isRunning;
    private volatile boolean isFinished;

    // Usado para disponibilizar tempo da captura das imagens
    private int imageCount = 0;
    private long totalTime = 0;
    private Font msgFont;

    // Variáveis do JavaCV
    private CvHaarClassifierCascade classifier;
    private CvMemStorage storage;
    private CanvasFrame debugCanvas;
    private IplImage grayIm;

    // Usado para as threads que executam a detecção das faces
    private ExecutorService executor;
    private AtomicInteger numTasks;

    // Usado para armazenar número de tarefas de detecção
    private long detectStartTime = 0;

    private String identificacao;           // Identificação da pessoa pelo LBPH 
    private IplImage[] facesTreinamento;    // Armazena as faces para treinamento do LBPH
    private LBPHFaceRecognizer lbphFaceRecognizer;
    private Rectangle faceRect;             // Armazena as coordenadas da face

    private volatile boolean saveFace = false;

    public FacePanel() {
        setBackground(Color.white);
        msgFont = new Font("SansSerif", Font.BOLD, 18);

        executor = Executors.newSingleThreadExecutor();
        /* O executor controla uma única thread com uma fila.
            Somente uma tarefa pode executar de cada vez. As outras devem esperar.
         */
        numTasks = new AtomicInteger(0);
        // Usado para limitar o tamanho da fila do executor.

        initDetector();
        faceRect = new Rectangle();
        lbphFaceRecognizer = new LBPHFaceRecognizer();
        facesTreinamento = new IplImage[lbphFaceRecognizer.getNUM_IMAGES_PER_PERSON()];

        isRunning = true;
        isFinished = false;

        new Thread(this).start();   // Atualiza as imagens para o painel.
    }

    private void initDetector() {
        // Cria uma instância do cascade classifier para detecção dos rostos
        classifier = new CvHaarClassifierCascade(cvLoad(FACE_CASCADE_FNM));
        if (classifier.isNull()) {
            System.out.println("\nCould not load the classifier file: " + FACE_CASCADE_FNM);
            System.exit(1);
        }

        storage = CvMemStorage.create();  //Cria armazenamento usado para detecção

        // debugCanvas = new CanvasFrame("Debugging Canvas");
        // useful for showing JavaCV IplImage objects, to check on image processing
    }  // end of initDetector()

    /*
        Deixa o painel no tamanho suficiente para uma imagem
    */
    public Dimension getPreferredSize()
    {
        return new Dimension(WIDTH, HEIGHT);
    }
    
    
    /*
        Exibe a imagem do dispositivo de captura a cada tempo de DELAY.
        A tarefa de detecção somente se inicia após cada tempo de DETECT_DELAY
        e somente se o número de tarefas no executor for menor que seu limite
    */
    public void run() {
        grabber = initGrabber(CAMERA_ID);
        if (grabber == null) {
            return;
        }

        long duration;

        while (isRunning) {
            long startTime = System.currentTimeMillis();

            snapIm = picGrab(grabber, CAMERA_ID);

            if (((System.currentTimeMillis() - detectStartTime) > DETECT_DELAY)
                    && (numTasks.get() < MAX_TASKS)) {
                trackFace(snapIm);
            }
            imageCount++;
            repaint();

            duration = System.currentTimeMillis() - startTime;
            totalTime += duration;

            if (duration < DELAY) {
                try {
                    Thread.sleep(DELAY - duration);  // Aguarda terminar o tempo de delay
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }

        }
        closeGrabber(CAMERA_ID);
        System.out.println("Execution End");
        isFinished = true;

    }
    
    
    /*
        Inicia objeto de captura
    */
    private FrameGrabber initGrabber(int ID) {
        FrameGrabber grabber = null;
        System.out.println("Inicializando captura pelo dispositivo: " + videoInput.getDeviceName(ID));
        try {
            grabber = FrameGrabber.createDefault(ID);
            grabber.setFormat("dshow");       // Usando DirectShow
            grabber.setImageWidth(WIDTH);     // tamanho padrão das imagens é pequeno: 320x240
            grabber.setImageHeight(HEIGHT);
            grabber.start();
        } catch (Exception e) {
            System.out.println("Could not start grabber");
            System.out.println(e);
            System.exit(1);
        }
        return grabber;
    }

    private IplImage picGrab(FrameGrabber grabber, int ID) {
        IplImage im = null;
        OpenCVFrameConverter.ToIplImage conversor = new OpenCVFrameConverter.ToIplImage();
        try {
            im = conversor.convert(grabber.grab());  //Tira uma foto
        } catch (Exception e) {
            System.out.println("Problem grabbing image for camera " + ID);
        }
        return im;
    }

    /*
        Finaliza a obtenção das imagens do dispositivo de câmera
    */    
    public void closeGrabber(int ID) {
        try {
            grabber.stop();
            grabber.release();
        } catch (Exception e) {
            System.out.println("Problem stopping grabbing for camera " + ID);
        }
    } 
    
    /*
        Desenha a imagem, o retângulo em volta a da face detectada e a média de tempo
        de obtenção das imagens da câmera no canto inferior esquerdo do painel.
        No canto superior exibe a face identificada.
        O tempo exibido não inclui a tarefa de detecção do rosto.
    */
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setFont(msgFont);

        // Desenha a imagem, estatísticas e retângulo
        if (snapIm != null) {
            g2.setColor(Color.YELLOW);
            OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
            Java2DFrameConverter paintConverter = new Java2DFrameConverter();
            org.bytedeco.javacv.Frame frame = grabberConverter.convert(snapIm);
            g2.drawImage(paintConverter.getBufferedImage(frame, 1), 0, 0, this);   // Desenha a imagem
            String statsMsg = String.format("Snap Avg. Time:  %.1f ms",
                    ((double) totalTime / imageCount));

            // Exibe a face identificada no canto superior esquerdo
            String box_text = "Identificado: " + identificacao;
            g2.drawString(box_text, 5, 50);

            // Escreve as estatísticas no canto inferior esquerdo
            g2.drawString(statsMsg, 5, HEIGHT - 10);
            

            drawRect(g2);
        } else {  // Caso ainda não obtiver nenhuma imagem
            g2.setColor(Color.BLUE);
            g2.drawString("Carregando câmera: " + CAMERA_ID, 5, HEIGHT - 10);
        }
    }

    
    /*
        Usa o retângulo da face a desenhar o retângulo verde em torno da face
        O desenho do retângulo está em um bloco sincronizado pois a variável faceRect pode estar sendo
        utilizada em outra thread.
    */
    private void drawRect(Graphics2D g2) {
        synchronized (faceRect) {
            if (faceRect.width == 0) {
                return;
            }

            // Desenha o retângulo
            g2.setColor(Color.GREEN);
            g2.setStroke(new BasicStroke(2));
            g2.drawRect(faceRect.x, faceRect.y, faceRect.width, faceRect.height);

        }
    }

    
    /*  Termina a execução do run() e espera a finalização do sistema.
        Para a aplicação até que sua execução termine.
    */
    public void closeDown() /* Terminate run() and wait for it to finish.
     This stops the application from exiting until everything
     has finished. */ {
        isRunning = false;
        while (!isFinished) {
            try {
                Thread.sleep(DELAY);
            } catch (Exception ex) {
            }
        }
    }

    
    
    
    
    
    // ------------------------- face tracking ----------------------------\\
    
    /*  Cria um thread para detectar faces nas imagens geradas pela câmera.
        Armazena as coordenadas das faces em faceRect e salva a imagem para treinamento caso solicitado.
        Imprime o tempo de execução no console
    */
    private void trackFace(IplImage img) {
        numTasks.getAndIncrement();     // Incrementa o número de tarefas antes de entrar na fila
        executor.execute(new Runnable() {
            public void run() {
                detectStartTime = System.currentTimeMillis();
                CvRect rect = findFace(img);
                if (rect != null) {

                    // Seta parâmetros do retângulo
                    setRectangle(rect);
                }
                long detectDuration = System.currentTimeMillis() - detectStartTime;
                System.out.println(" detection duration: " + detectDuration + "ms");
                numTasks.getAndDecrement();  // Decrementa o número de tarefas quando terminado
            }
        });
    }

    
    /*  Formata a imagem e converte para escala de cinza. Formatação deixa a imagem menor
        tornando o processamento mais rápido.
        O detector Haar precisa como parâmetro a imagem em escala de cinza    
    */
    private IplImage scaleGray(IplImage img) {
        // Converte para escalas de cinza
        IplImage grayImg = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
        cvCvtColor(img, grayImg, CV_BGR2GRAY);

        // Formata a imagem
        IplImage smallImg = IplImage.create(grayImg.width() / IM_SCALE,
                grayImg.height() / IM_SCALE, IPL_DEPTH_8U, 1);
        cvResize(grayImg, smallImg, CV_INTER_LINEAR);

        // Equaliza a imagem menor em escalas de cinza
        cvEqualizeHist(smallImg, smallImg);
        return smallImg;
    } 

    
    /*  Detecta somente uma face utilizando o Haar Detector
        Chama a função para a identificação do rosto encontrado.
    */
    private CvRect findFace(IplImage img) 
    {
        // Converte para escalas de Cinza
        grayIm = scaleGray(img);

        /*
     // Mostra a imagem em escalas de cinza para verificar o processo de tratamento da imagem
     debugCanvas.showImage(grayIm);
	 debugCanvas.waitKey(0);
         */
        // System.out.println("Detecting largest face...");   // cvImage
        CvSeq faces = cvHaarDetectObjects(grayIm, classifier, storage, 1.1, 1, // 3
                // CV_HAAR_SCALE_IMAGE |
                CV_HAAR_DO_ROUGH_SEARCH | CV_HAAR_FIND_BIGGEST_OBJECT);
        // Por questões de segurança e rapidez, só será detectado o rosto maior e mais próximo.

        int total = faces.total();
        if (total == 0) {
            System.out.println("No faces found");
            return null;
       } else if (total > 1) //Este caso não deveria ocorrer. Incluído por segurança
        {
            System.out.println("Multiple faces detected (" + total + "); using the first");
        } else {
            System.out.println("Face detected");
        }

        CvRect rect = new CvRect(cvGetSeqElem(faces, 0));

        if (saveFace) {
            learnNewFace(img);
        }

        //Identifica o rosto
        identificacao = lbphFaceRecognizer.identifyFace(grayIm);

        cvClearMemStorage(storage);
        return rect;
    }

    
    
    /* Extrai o tamanho e as coordenadas da imagem desatacada da estrutura do retângulo do JavaCV
       e armazena em um retângulo Java.
       Durante o processo, desfaz o escalonamento que foi aplicado a imagem antes da detecção do rosto.
       Disponibiliza informações de movimento do rosto na imagem.
       O uso dessa função está em um bloco sincronizado pois o retângulo pode estar sendo utilizado para
       atualizar o painel de imagem ou pintura do retângulo ao mesmo tempo em outras threads.
    */
    private void setRectangle(CvRect r) {
        synchronized (faceRect) {
            int xNew = r.x() * IM_SCALE;
            int yNew = r.y() * IM_SCALE;
            int widthNew = r.width() * IM_SCALE;
            int heightNew = r.height() * IM_SCALE;

            // calcula o movimento do retângulo comparado com o anterior
            int xMove = (xNew + widthNew / 2) - (faceRect.x + faceRect.width / 2);
            int yMove = (yNew + heightNew / 2) - (faceRect.y + faceRect.height / 2);

            // Dispõe informações de movimento da face se for significante
            if ((Math.abs(xMove) > SMALL_MOVE) || (Math.abs(yMove) > SMALL_MOVE)) {
                System.out.println("Movement (x,y): (" + xMove + "," + yMove + ")");
            }

            faceRect.setRect(xNew, yNew, widthNew, heightNew);
        }
    }

    // --------------------------- Salvar e Aprender nova Face -----------------------------------
    public void saveFace(String faceID) {
        saveFace = true;
        FACE_ID = faceID;
    }

    /* Armazena e aprende nova Face */
    private void learnNewFace(IplImage img) {
        numTasks.getAndIncrement();         // Incrementa número de tarefas antes de entrar na fila.
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    for (int i = 0; i < lbphFaceRecognizer.getNUM_IMAGES_PER_PERSON(); i++) {
                        facesTreinamento[i] = img; //clipSaveFace(img);
                    }
                    lbphFaceRecognizer.saveNewFace(FACE_ID, facesTreinamento);
                    saveFace = false;
                } catch (Exception ex) {
                    Logger.getLogger(FacePanel.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        });
    }

    /* Corta a imagem utilizando do quadrado que possui as coordenadas da face
     O uso dessa função está em um bloco sincronizado pois o retângulo pode estar sendo utilizado para
     atualizar o painel de imagem ou pintura do retângulo ao mesmo tempo em outras threads.
     
     */
    private IplImage clipSaveFace(IplImage img) {
        BufferedImage clipIm = null;

        synchronized (faceRect) {
            if (faceRect.width == 0) {
                System.out.println("No face selected");
                return img;
            }
            OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
            Java2DFrameConverter paintConverter = new Java2DFrameConverter();
            org.bytedeco.javacv.Frame frame = grabberConverter.convert(img);
            BufferedImage im = paintConverter.getBufferedImage(frame, 1);
            try {
                clipIm = im.getSubimage(faceRect.x, faceRect.y, faceRect.width, faceRect.height);
            } catch (RasterFormatException e) {
                System.out.println("Could not clip the image");
            }
        }
        if (clipIm != null) {
            BufferedImage grayIm = resizeImage(clipIm);
            BufferedImage faceIm = clipToFace(grayIm);
            return toIplImage(faceIm);
        } else {
            return img;
        }
    }

    /* Altera a imagem para um tamanho padrão e a transforma em escalas de cinza */
    private BufferedImage resizeImage(BufferedImage im) {
        int imWidth = im.getWidth();
        int imHeight = im.getHeight();
        System.out.println("Original (w,h): (" + imWidth + ", " + imHeight + ")");

        double widthScale = FACE_WIDTH / ((double) imWidth);
        double heightScale = FACE_HEIGHT / ((double) imHeight);
        double scale = (widthScale > heightScale) ? widthScale : heightScale;

        int nWidth = (int) Math.round(imWidth * scale);
        int nHeight = (int) Math.round(imHeight * scale);

        // Converte para escalas de cinza enquanto muda seu tamanho
        BufferedImage grayIm = new BufferedImage(nWidth, nHeight,
                BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2 = grayIm.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2.drawImage(im, 0, 0, nWidth, nHeight, 0, 0, imWidth, imHeight, null);
        g2.dispose();

        System.out.println("Scaled gray (w,h): (" + nWidth + ", " + nHeight + ")");
        return grayIm;
    }
        

    /* Corta a imagem em no tamanho FACE_WIDTHxFACE-HEIGHT
       Assume que a imagem do parâmetro é do tamanho da face ou maior
     */
    private BufferedImage clipToFace(BufferedImage im) {
        int xOffset = (im.getWidth() - FACE_WIDTH) / 2;
        int yOffset = (im.getHeight() - FACE_HEIGHT) / 2;
        BufferedImage faceIm = null;
        try {
            faceIm = im.getSubimage(xOffset, yOffset, FACE_WIDTH, FACE_HEIGHT);
            System.out.println("Clipped image to face dimensions: ("
                    + FACE_WIDTH + ", " + FACE_HEIGHT + ")");
        } catch (RasterFormatException e) {
            System.out.println("Could not clip the image");
            OpenCVFrameConverter.ToIplImage grabberConverter = new OpenCVFrameConverter.ToIplImage();
            Java2DFrameConverter paintConverter = new Java2DFrameConverter();
            org.bytedeco.javacv.Frame frame = grabberConverter.convert(grayIm);
            faceIm = paintConverter.getBufferedImage(frame);
        }
        return faceIm;
    }
        
        
    /* Converte uma BufferedImage para IplImage para utilização do OpenCV */
    
    private IplImage toIplImage(BufferedImage bufImage) {
        ToIplImage iplConverter = new OpenCVFrameConverter.ToIplImage();
        Java2DFrameConverter java2dConverter = new Java2DFrameConverter();
        IplImage iplImage = iplConverter.convert(java2dConverter.convert(bufImage));
        return iplImage;
    }

    public processing.LBPHFaceRecognizer getLbphFaceRecognizer() {
        return lbphFaceRecognizer;
    }

    public void setLbphFaceRecognizer(processing.LBPHFaceRecognizer lbphFaceRecognizer) {
        this.lbphFaceRecognizer = lbphFaceRecognizer;
    }

} // end of FacePanel class

