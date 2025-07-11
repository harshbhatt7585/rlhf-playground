'use client';

import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const NeuralNetworkVisualization: React.FC = () => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animationIdRef = useRef<number | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Scene setup
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
    camera.position.set(0, 0, 3);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lights
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
    scene.add(hemiLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 10, 7);
    scene.add(dirLight);

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.1;
    controls.autoRotate = false;
    controlsRef.current = controls;

    // Resize handler
    const onResize = (): void => {
      if (!container || !camera || !renderer) return;
      const width = container.clientWidth * 2;
      const height = container.clientHeight * 2;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    window.addEventListener('resize', onResize);
    onResize();

    // Build network
    const layers = [4, 6, 6, 4, 1];
    const spacing = 1;
    const nodeGeo = new THREE.SphereGeometry(0.07, 16, 16);

    const networkGroup = new THREE.Group();
    scene.add(networkGroup);

    const nodes: THREE.Mesh[] = [];
    layers.forEach((count, layerIdx) => {
      for (let i = 0; i < count; i++) {
        const mat = new THREE.MeshStandardMaterial({
          color: new THREE.Color().setHSL(layerIdx / layers.length, 0.7, 0.6),
          roughness: 0.4,
          metalness: 0.5,
          transparent: true,
          opacity: 0.8,
        });
        const mesh = new THREE.Mesh(nodeGeo, mat);
        mesh.position.set(
          (layerIdx - (layers.length - 1) / 2) * spacing,
          (i - (count - 1) / 2) * 0.5,
          0
        );
        networkGroup.add(mesh);
        nodes.push(mesh);
      }
    });

    const connectionGroup = new THREE.Group();
    scene.add(connectionGroup);
    const lineMat = new THREE.LineBasicMaterial({ transparent: true, opacity: 0.2 });

    nodes.forEach((node, idx) => {
      const layer = Math.round(node.position.x / spacing + (layers.length - 1) / 2);
      if (layer < layers.length - 1) {
        const prevCountSum = layers.slice(0, layer).reduce((a, b) => a + b, 0);
        const nextCountSum = layers.slice(0, layer + 1).reduce((a, b) => a + b, 0);
        const nextNodes = nodes.slice(nextCountSum, nextCountSum + layers[layer + 1]);
        nextNodes.forEach(target => {
          if (Math.random() < 0.6) {
            const geom = new THREE.BufferGeometry().setFromPoints([
              node.position.clone(),
              target.position.clone(),
            ]);
            const line = new THREE.Line(geom, lineMat.clone());
            connectionGroup.add(line);
          }
        });
      }
    });

    // Animate
    const animate = (): void => {
      animationIdRef.current = requestAnimationFrame(animate);
      const t = Date.now() * 0.001;

      nodes.forEach((mesh, i) => {
        const pulse = Math.sin(t * 5 + i) * 0.5 + 0.5;
        mesh.scale.setScalar(1 + pulse * 0.2);
        mesh.material.opacity = 0.5 + pulse * 0.3;
      });

      connectionGroup.children.forEach((line, i) => {
        const mat = (line as THREE.Line).material as THREE.LineBasicMaterial;
        mat.opacity = 0.1 + Math.sin(t * 4 + i * 0.1) * 0.1;
      });

      controls?.update();
      renderer?.render(scene, camera);
    };
    animate();

    return () => {
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
      window.removeEventListener('resize', onResize);
      controls.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  return <div ref={containerRef} className="flex justify-center items-center w-full h-full" />;
};

export default NeuralNetworkVisualization;