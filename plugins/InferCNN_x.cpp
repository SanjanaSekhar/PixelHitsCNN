/*
 * Example plugin to demonstrate the direct multi-threaded inference on CNN with TensorFlow 2.
 */

#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <cfloat>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <vector>
#include <iomanip>
#include <sstream>
#include <fstream>

#include <TF1.h>
#include "Math/MinimizerOptions.h"
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TMath.h>
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TObject.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TStyle.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TPostScript.h"
#include "Math/DistFunc.h"
#include "TTree.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/SiPixelTemplateReco.h"
#include "CalibTracker/Records/interface/SiPixelTemplateDBObjectESProducerRcd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
class TObject;
class TTree;
class TH1D;
class TFile;

using namespace std;
using namespace edm;
using namespace reco;
// define the cache object
// it could handle graph loading and destruction on its own,
// but in this example, we define it as a logicless container
struct CacheData {
	CacheData() : graphDef(nullptr) {}
	std::atomic<tensorflow::GraphDef*> graphDef;
};

class InferCNN_x : public edm::stream::EDAnalyzer<edm::GlobalCache<CacheData>> {
public:
	explicit InferCNN_x(const edm::ParameterSet&, const CacheData*);
	~InferCNN_x(){};

	static void fillDescriptions(edm::ConfigurationDescriptions&);

	// two additional static methods for handling the global cache
	static std::unique_ptr<CacheData> initializeGlobalCache(const edm::ParameterSet&);
	static void globalEndJob(const CacheData*); // does it have to be static

	private:
		void beginJob();
		void analyze(const edm::Event&, const edm::EventSetup&);
		void endJob();

		std::string inputTensorName_x, inputTensorName_y, anglesTensorName_x, anglesTensorName_y;
		std::string outputTensorName_;
	//std::string     fRootFileName;
		tensorflow::Session* session_x;
		TFile *fFile; TTree *fTree;
		static const int MAXCLUSTER = 100000;
		float x_gen[MAXCLUSTER], x_1dcnn[MAXCLUSTER], dx[MAXCLUSTER]; 
		int count; char path[100], infile1[100], infile2[100], infile3[300];
		edm::InputTag fTrackCollectionLabel, fPrimaryVertexCollectionLabel;
		std::string     fRootFileName;
		edm::EDGetTokenT<std::vector<reco::Track>> TrackToken;
		edm::EDGetTokenT<reco::VertexCollection> VertexCollectionToken;
		FILE *cnn_file, *gen_file;
	//const bool applyVertexCut_;

	//edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
	//edm::EDGetTokenT<reco::VertexCollection> offlinePrimaryVerticesToken_;
	//edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
	//edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
	};

	std::unique_ptr<CacheData> InferCNN_x::initializeGlobalCache(const edm::ParameterSet& config) 
	{

	// this method is supposed to create, initialize and return a CacheData instance
		CacheData* cacheData = new CacheData();

	// load the graph def and save it
		std::string graphPath_x = config.getParameter<std::string>("graphPath_x");
		cacheData->graphDef = tensorflow::loadGraphDef(graphPath_x);

	// set tensorflow log leven to warning
		tensorflow::setLogging("2");
	//init();

		return std::unique_ptr<CacheData>(cacheData);
	}

	void InferCNN_x::globalEndJob(const CacheData* cacheData) {
	// reset the graphDef
		if (cacheData->graphDef != nullptr) {
			delete cacheData->graphDef;
		}

	}

	void InferCNN_x::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
	// defining this function will lead to a *_cfi file being generated when compiling
		edm::ParameterSetDescription desc;
		desc.add<std::string>("graphPath_x");
		desc.add<std::string>("inputTensorName_x");
		desc.add<std::string>("anglesTensorName_x");
		desc.add<std::string>("outputTensorName");
		descriptions.addWithDefaultLabel(desc);
	}

	InferCNN_x::InferCNN_x(const edm::ParameterSet& config, const CacheData* cacheData)
	: inputTensorName_x(config.getParameter<std::string>("inputTensorName_x")),
	anglesTensorName_x(config.getParameter<std::string>("anglesTensorName_x")),
	outputTensorName_(config.getParameter<std::string>("outputTensorName")),
	session_x(tensorflow::createSession(cacheData->graphDef)),
//fVerbose(config.getUntrackedParameter<int>("verbose", 0)),
//fTrackCollectionLabel(config.getUntrackedParameter<InputTag>("trackCollectionLabel", edm::InputTag("ALCARECOTkAlMuonIsolated"))),
	fTrackCollectionLabel(config.getUntrackedParameter<InputTag>("trackCollectionLabel", edm::InputTag("generalTracks"))),
	fPrimaryVertexCollectionLabel(config.getUntrackedParameter<InputTag>("PrimaryVertexCollectionLabel", edm::InputTag("offlinePrimaryVertices"))),
	fRootFileName(config.getUntrackedParameter<string>("rootFileName", string("x_1dcnn.root"))) {

		TrackToken              = consumes <std::vector<reco::Track>>(fTrackCollectionLabel) ;
		VertexCollectionToken   = consumes <reco::VertexCollection>(fPrimaryVertexCollectionLabel) ;
		count = 0;

	//initializations
		for(int i=0;i<MAXCLUSTER;i++){
			x_1dcnn[i]=-999.0;
			x_gen[i]=-999.0;
			dx[i]=-999.0;
			}
			sprintf(path,"TrackerStuff/PixelHitsCNN/txt_files");

			sprintf(infile1,"generic_MC_x.txt");
	//		gen_file = fopen(infile1, "w");

			sprintf(infile2,"1dcnn_MC_x.txt");
	//		cnn_file = fopen(infile2, "w");

			sprintf(infile3,"%s/cnn_MC_x.txt",path);
			cnn_file = fopen(infile3, "w");
		
	}

	void InferCNN_x::beginJob() {
	/*
	printf("IN BEGINJOB");
	fFile = TFile::Open(fRootFileName.c_str(), "RECREATE");
	fFile->cd();
	fTree = new TTree("x_rec", "x_rec");
 // fTree->Branch("x_gen",        x_gen,       "x_gen");
	fTree->Branch("x_1dcnn",       x_1dcnn,       "x_1dcnn/F");
	fTree->Branch("x_gen",        x_gen,       "x_gen/F");
	fTree->Branch("dx_1dcnn",       dx,       "dx_1dcnn/F");
	*/
	}

	void InferCNN_x::endJob() {
	// close the session
		tensorflow::closeSession(session_x);
				
		fclose(cnn_file);
	//	fclose(cnn_file);
		//fclose(res_gen_1cnn_file);
	/*
	//fTree->Fill();
	fFile->cd();
	fTree->Write();
	fFile->Write();
	fFile->Close();
	//delete fFile;
	printf("IN ENDJOB");
	*/
	}

	void InferCNN_x::analyze(const edm::Event& event, const edm::EventSetup& setup) {


//		if (gen_file==NULL) {
//			printf("couldn't open generic output file/n");
//			return ;
//		}
	//	if (cnn_file==NULL) {
	//		printf("couldn't open cnn output file/n");
	//		return ;
	//	}
		if (cnn_file==NULL) {
			printf("couldn't open residual output file/n");
			return ;
		}
		// get geometry
	/*
	edm::ESHandle<TrackerGeometry> tracker = setup.getHandle(trackerGeomToken_);
	assert(tracker.isValid());

	edm::ESHandle<TrackerTopology> tTopoHandle = setup.getHandle(trackerTopoToken_);
	auto const& tkTpl = *tTopoHandle;

	edm::Handle<reco::VertexCollection> vertices;
	if (applyVertexCut_) {
		event.getByToken(offlinePrimaryVerticesToken_, vertices);
		if (!vertices.isValid() || vertices->empty())
			return;
	}
	*/

	//TH1F* res_x = new TH1F("h706","dx = x_gen - x_1dcnn (all sig)",120,-300,300);

	//get the map
		edm::Handle<reco::TrackCollection> tracks;
	//event.getByToken(TrackToken, tracks);
	//int nTk(0);

		try {
			event.getByToken(TrackToken, tracks);
		}catch (cms::Exception &ex) {
	//if (fVerbose > 1) 
			cout << "No Track collection with label " << fTrackCollectionLabel << endl;
		}
		if (tracks.isValid()) {
			const std::vector<reco::Track> trackColl = *(tracks.product());
		//nTk = trackColl.size();
		//if (fVerbose > 1) 
		//cout << "--> Track collection size: " << nTk << endl;
		} else {
  	//if (fVerbose > 1)
			cout << "--> No valid track collection" << endl;
		}
		if (!tracks.isValid()) {
			cout << "track collection is not valid" <<endl;
			return;
		}

	//printf("Track is valid\n");
	//printf("Track collection size: %d\n",tracks->size());
	//stuff needed for template
		float clusbuf[TXSIZE][TYSIZE];
	//int mrow=TXSIZE,mcol=TYSIZE;
	//static float xrec, yrec;
		static int ix,iy;
		int prev_count = count;
		for (auto const& track : *tracks) {
		//if (applyVertexCut_ &&
		//	(track.pt() < 0.75 || std::abs(track.dxy((*vertices)[0].position())) > 5 * track.dxyError()))
		//	continue;

		//bool isBpixtrack = false, isFpixtrack = false, crossesPixVol = false;

		// find out whether track crosses pixel fiducial volume (for cosmic tracks)
		//auto d0 = track.d0(), dz = track.dz();
		//if (std::abs(d0) < 16 && std::abs(dz) < 50)
		//crossesPixVol = true;

			auto etatk = track.eta();

			auto const& trajParams = track.extra()->trajParams();
			assert(trajParams.size() == track.recHitsSize());
			auto hb = track.recHitsBegin();

			for (unsigned int h = 0; h < track.recHitsSize(); h++) {
				auto hit = *(hb + h);
				if (!hit->isValid())
					continue;
				auto id = hit->geographicalId();

			// check that we are in the pixel detector
				auto subdetid = (id.subdetId());
			//if (subdetid == PixelSubdetector::PixelBarrel)
			//isBpixtrack = true;
			//if (subdetid == PixelSubdetector::PixelEndcap)
			//isFpixtrack = true;
				if (subdetid != PixelSubdetector::PixelBarrel && subdetid != PixelSubdetector::PixelEndcap)
					continue;
				bool iAmBarrel = subdetid == PixelSubdetector::PixelBarrel;

			// PXB_L4 IS IN THE OTHER WAY
			// CAN BE XORed BUT LETS KEEP THINGS SIMPLE
			//	bool iAmOuter = ((tkTpl.pxbLadder(id) % 2 == 1) && tkTpl.pxbLayer(id) != 4) ||
			//((tkTpl.pxbLadder(id) % 2 != 1) && tkTpl.pxbLayer(id) == 4);

				auto pixhit = dynamic_cast<const SiPixelRecHit*>(hit->hit());
				if (!pixhit)
					continue;


			//some initialization
				for(int j=0; j<TXSIZE; ++j) {for(int i=0; i<TYSIZE; ++i) {clusbuf[j][i] = 0.;} } 


			// get the cluster
					auto clustp = pixhit->cluster();
				if (clustp.isNull())
					continue;
				auto const& cluster = *clustp;
				const std::vector<SiPixelCluster::Pixel> pixelsVec = cluster.pixels();

				auto const& ltp = trajParams[h];

				float cotAlpha=ltp.dxdz();
				float cotBeta=ltp.dydz();
//=======================================================================================
			/*
			printf("cluster.minPixelRow() = %i\n",cluster.minPixelRow());
			printf("cluster.minPixelCol() = %i\n",cluster.minPixelCol());

			int minPixelRow = 161;
			int minPixelCol = 417;

			float tmp_x = float(minPixelRow) + 0.5f;
 			float tmp_y = float(minPixelCol) + 0.5f;

 			
			printf("pixelsVec.size() = %lu\n",pixelsVec.size());
			// fix issues with the coordinate system: extract the centre pixel and its coords
			for (unsigned int i = 0; i < pixelsVec.size(); ++i) {
					float pixx = pixelsVec[i].x;  // index as float=iteger, row index
					float pixy = pixelsVec[i].y;  // same, col index

					printf("pixelsVec[%i].adc = %i, pixx = %f, pixy = %f\n",i,pixelsVec[i].adc,pixx,pixy);

					
					//  Find lower left corner pixel and its coordinates
					if((int)pixx < minPixelRow) {
						minPixelRow = (int)pixx; 
					}
					if((int)pixy < minPixelCol) {
						minPixelCol = (int)pixy;
					}
				}  // End loop over pixels


				// Now fill the cluster buffer with charges
				for (unsigned int i = 0; i < pixelsVec.size(); ++i) {

					float pixx = pixelsVec[i].x;  // index as float=iteger, row index
					float pixy = pixelsVec[i].y;  // same, col index
					float pixel_charge = pixelsVec[i].adc;

					ix = (int)pixx - minPixelRow;
					if(ix >= TXSIZE) continue;
					iy = (int)pixy - minPixelCol;
					if(iy >= TYSIZE) continue;

					//skip over double pixels: CHECK THIS

					if ((int)pixx == 79){
						i+=2; continue;
					}
					if ((int)pixy % 52 == 51 ){
						i+=2; continue; 
					}
					clusbuf[ix][iy] = pixel_charge;
					printf("ix = %i, iy = %i\n",ix,iy);
				}
				*/
//========================================================================================
				  // Preparing to retrieve ADC counts from the SiPixeltheClusterParam.theCluster->  In the cluster,
  // we have the following:
  //   int minPixelRow(); // Minimum pixel index in the x direction (low edge).
  //   int maxPixelRow(); // Maximum pixel index in the x direction (top edge).
  //   int minPixelCol(); // Minimum pixel index in the y direction (left edge).
  //   int maxPixelCol(); // Maximum pixel index in the y direction (right edge).
  // So the pixels from minPixelRow() will go into clust_array_2d[0][*],
  // and the pixels from minPixelCol() will go into clust_array_2d[*][0].

				int row_offset = cluster.minPixelRow();
				int col_offset = cluster.minPixelCol();
//			printf("cluster.minPixelRow() = %i\n",cluster.minPixelRow());
//			printf("cluster.minPixelCol() = %i\n",cluster.minPixelCol());
  // Store the coordinates of the center of the (0,0) pixel of the array that
  // gets passed to PixelTempReco1D
  // Will add these values to the output of  PixelTempReco1D
				float tmp_x = float(row_offset) + 0.5f;
				float tmp_y = float(col_offset) + 0.5f;
//			printf("tmp_x = %f, tmp_y = %f\n", tmp_x,tmp_y);

//			printf("cluster.size() = %i\n",cluster.size());

				  // first compute matrix size
				int mrow = 0, mcol = 0;
				for (int i = 0; i != cluster.size(); ++i) {
					auto pix = cluster.pixel(i);
					int irow = int(pix.x);
					int icol = int(pix.y);
					mrow = std::max(mrow, irow);
					mcol = std::max(mcol, icol);
				}
				mrow -= row_offset;
				mrow += 1;
				mrow = std::min(mrow, TXSIZE);
				mcol -= col_offset;
				mcol += 1;
				mcol = std::min(mcol, TYSIZE);
				assert(mrow > 0);
				assert(mcol > 0);
//			printf("mrow = %i, mcol = %i\n",mrow,mcol);
  //float clusbuf[mrow][mcol];
  //memset(clusbuf, 0, sizeof(float) * mrow * mcol);
				int clustersize_x = 0, clustersize_y = 0;
				int same_x = 500, same_y = 500; //random initial value
				for (int i = 0; i < cluster.size(); ++i) {
					auto pix = cluster.pixel(i);
					int irow = int(pix.x) - row_offset;
					int icol = int(pix.y) - col_offset;
					//double pixels skip
					if ((int)pix.x == 79){
						i+=2; continue;
					}
					if ((int)pix.y % 52 == 51 ){
						i+=2; continue; 
					}

					if(irow != same_x){
						clustersize_x++;
						same_x = irow;
					}
					if(icol != same_y){
						clustersize_y++;
						same_y = icol;
					}
				}

				int mid_x = int(clustersize_x/2)-1;
				int mid_y = int(clustersize_y/2)-1;
				int offset_x = 6 - mid_x;
				int offset_y = 10 - mid_y;

  // Copy clust's pixels (calibrated in electrons) into clusMatrix;
				for (int i = 0; i < cluster.size(); ++i) {
					auto pix = cluster.pixel(i);
					int irow = int(pix.x) - row_offset + offset_x;
					int icol = int(pix.y) - col_offset + offset_y;

    // Gavril : what do we do here if the row/column is larger than cluster_matrix_size_x/cluster_matrix_size_y  ?
    // Ignore them for the moment...
					if ((int)pix.x == 79){
						i+=2; continue;
					}
					if ((int)pix.y % 52 == 51 ){
						i+=2; continue; 
					}
					if ((irow > mrow) || (icol > mcol)) continue;
					clusbuf[irow][icol] = float(pix.adc);
 //   printf("pix[%i].adc = %i, pix.x = %i, pix.y = %i, irow = %i, icol = %i\n",i,pix.adc,pix.x,pix.y,irow,icol);

				}
//			printf("fails after filling buffer\n");
 			//https://github.com/cms-sw/cmssw/blob/master/RecoLocalTracker/SiPixelRecHits/src/PixelCPEBase.cc#L263-L272
				LocalPoint trk_lp = ltp.position();
				float trk_lp_x = trk_lp.x();
				float trk_lp_y = trk_lp.y();

				Topology::LocalTrackPred loc_trk_pred =Topology::LocalTrackPred(trk_lp_x, trk_lp_y, cotAlpha, cotBeta);
				LocalPoint lp; 
				auto geomdetunit = dynamic_cast<const PixelGeomDetUnit*>(pixhit->detUnit());
				auto const& topol = geomdetunit->specificTopology();
				lp = topol.localPosition(MeasurementPoint(tmp_x, tmp_y), loc_trk_pred);
		//	printf("fails after lp\n");
				//===============================
				// define a tensor and fill it with cluster projection
				tensorflow::Tensor cluster_flat_x(tensorflow::DT_FLOAT, {1,TXSIZE,1});
    		// angles
				tensorflow::Tensor angles(tensorflow::DT_FLOAT, {1,2});
				angles.tensor<float,2>()(0, 0) = cotAlpha;
				angles.tensor<float,2>()(0, 1) = cotBeta;

				for (size_t i = 0; i < TXSIZE; i++) {
					cluster_flat_x.tensor<float,3>()(0, i, 0) = 0;
					for (size_t j = 0; j < TYSIZE; j++){
            //1D projection in x
						cluster_flat_x.tensor<float,3>()(0, i, 0) += clusbuf[i][j];
						printf("%f ",clusbuf[i][j]);

					}
					printf("\n");
				}

				// TODO: CENTER THE CLUSTER


				// define the output and run
				std::vector<tensorflow::Tensor> output_x;
				tensorflow::run(session_x, {{inputTensorName_x,cluster_flat_x}, {anglesTensorName_x,angles}}, {outputTensorName_}, &output_x);
				// convert microns to cms
				x_1dcnn[count] = output_x[0].matrix<float>()(0,0)*1.0e-4; 
				// go back to module coordinate system
				x_1dcnn[count]+=lp.x(); 
				// get the generic position
				x_gen[count] = hit->localPosition().x();

				// compute the residual
				dx[count] = x_gen[count] - x_1dcnn[count];
//			printf("Generic position: %f\n ",x_gen[count]*1e4);
//			printf("1dcnn position: %f\n ",x_1dcnn[count]*1e4);
//			printf("%i\n",count);
				count++;

			}
		}
//	printf("count = %i\n",count);
		//fTree->Fill();
		//printf("Output from generic:\n");
	//	for(int i=prev_count;i<count;i++){
	//		printf("%f\n", x_gen[i]);
	//	}
	//	printf("Output from 1dcnn:\n");
	//	for(int i=prev_count;i<count;i++){
	//		printf("%f\n", x_1dcnn[i]);
	//	}
	//	printf("dx residual:\n");
		for(int i=prev_count;i<count;i++){
			fprintf(cnn_file,"%f %f\n", x_gen[i],x_1dcnn[i]);
		}

	}
	DEFINE_FWK_MODULE(InferCNN_x);