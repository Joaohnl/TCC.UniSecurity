package gui;

// FaceTracker.java
// Andrew Davison, July 2013, ad@fivedots.psu.ac.th

/* Show a sequence of images snapped from a webcam in a picture panel (FacePanel). 
   A face is highlighted with a yellow rectangle, which is updated as the face
   moves. The highlighted part of the image can be saved by the user pressing
   the "Save Face" button.

   Usage:
      > java FaceTracker
 */
import java.awt.*;
import java.awt.event.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.*;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_objdetect;
import processing.FacePanel;
import processing.LBPHFaceRecognizer;

public class telaPrincipal extends JFrame {
    // GUI components

    private FacePanel facePanel;

    public telaPrincipal() {
        super("UniSecurity");

        Container c = getContentPane();
        c.setLayout(new BorderLayout());

        // Preload the opencv_objdetect module to work around a known bug.
        Loader.load(opencv_objdetect.class);
        
        facePanel = new FacePanel(); // the sequence of pictures appear here
        c.add(facePanel, BorderLayout.CENTER);
        

        // Botão para salvar face detectada
        JButton btnSalvarFace = new JButton("Salvar Face");
        JButton btnTreinarModelo = new JButton("Treinar Modelo");
        
        JTextField txtIdentificacao = new JTextField();
        txtIdentificacao.setText("Insira o RA do aluno");
        
        //Limpa o campo de texto assim que é clicado
        txtIdentificacao.addMouseListener(new MouseAdapter(){           
            @Override
            public void mouseClicked(MouseEvent e){
                txtIdentificacao.setText("");
            }
        });
        
        
        txtIdentificacao.addKeyListener(new KeyAdapter() {
            // Habilita ou desabilita o botão de salvar face.
            // Campo de ter 9 digítos (tamanho do RA dos alunos) para habilitar o botão.
            @Override
            public void keyReleased(KeyEvent e) { // Observa entrada de dígitos no campo
                try {
                    int teste = Integer.parseInt(txtIdentificacao.getText());
                    if (txtIdentificacao.getText().length() != 9) {
                        btnSalvarFace.setEnabled(false);
                    } else {
                        btnSalvarFace.setEnabled(true);
                    }
                } catch (NumberFormatException exc) {
                    txtIdentificacao.setText("");
                }

            }
        });
        
        btnSalvarFace.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                facePanel.saveFace(txtIdentificacao.getText());
            }
        });
        
        btnSalvarFace.setEnabled(false);

        JPanel p = new JPanel();
        p.add(txtIdentificacao);
        p.add(btnSalvarFace);
        c.add(p, BorderLayout.SOUTH);

        addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                facePanel.closeDown();    // stop snapping pics
                System.exit(0);
            }
        });

        setResizable(false);
        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }
} // end of FaceTracker class
